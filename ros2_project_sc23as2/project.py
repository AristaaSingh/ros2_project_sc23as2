import threading
import cv2
import numpy as np
import rclpy
import math
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav2_msgs.action import NavigateToPose
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
from math import sin, cos
import signal


# The contour area (in pixels) of the blue blob when the robot is approximately
# 1 metre away from the blue box. When the blob reaches this size the robot stops.
# This value needs tuning — run the code, drive close to the box, and read the
# logged area value. Set this to whatever area you see at ~1 m distance.
BLUE_STOP_AREA = 30000

# The list of (x, y, theta) map coordinates the robot will navigate to one by one.
# x, y  — position on the map in metres
# theta — the direction the robot faces when it arrives, in radians
#          0.0   = facing right (+x)
#          1.57  = facing up (+y)
#          3.14  = facing left (-x)
#         -1.57  = facing down (-y)
# These coordinates were found using the Publish Point tool in RViz.
# Adjust the theta values so the camera faces toward the interior of the room
# at each stop — that way the boxes are more likely to be in view.
WAYPOINTS = [
    (-3.0, -3.0,  0.0),
    (-1.0,  -5.0,  math.pi),    # centre, facing right
    ( 1.65, -11.0, math.pi/2),   # lower area, facing up into the room
]


class ColourIdentifier(Node):

    def __init__(self):
        super().__init__('colour_identifier')

        # CvBridge converts ROS image messages into OpenCV images we can process
        self.bridge = CvBridge()

        # Sensitivity controls how wide the colour detection range is in HSV.
        # A value of 10 means we accept hues within +/- 10 of the target hue.
        self.sensitivity = 10

        # These flags are set to True inside the camera callback whenever
        # each colour is detected in the current frame
        self.red_found   = False
        self.green_found = False
        self.blue_found  = False

        # These flags are set to True the first time each colour is ever detected
        # during the run. They stay True even if the colour goes out of view later.
        # Used to print the detection messages and the final summary.
        self.green_detected = False
        self.red_detected   = False
        self.blue_detected  = False

        # Ensures the final colour summary is only printed once
        self.summary_printed = False

        # The pixel area of the blue blob — used to decide when to stop approaching
        self.blue_area = 0

        # The horizontal pixel position of the centre of the blue blob.
        # The image is 320 pixels wide so 160 is the centre.
        # We use this to steer left or right toward the blue box.
        self.blue_pos = 160

        # The robot uses a simple state machine to decide what to do.
        # States: 'exploring', 'approaching', 'stopped'
        self.state = 'exploring'

        # waypoint_idx tracks which waypoint to send to Nav2 next
        self.waypoint_idx = 0

        # goal_in_flight is True while Nav2 is actively navigating to a waypoint
        self.goal_in_flight = False

        # nav_goal_handle stores the handle of the current Nav2 goal
        # so we can cancel it mid-journey if the blue box is spotted
        self.nav_goal_handle = None

        # Publisher to send velocity commands directly to the robot motors.
        # This is used for approaching and rotating — not during Nav2 navigation.
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # The Nav2 action client lets us send navigation goals to Nav2.
        # Nav2 then plans a path and drives the robot there autonomously.
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Subscribe to the camera topic. Every time a new image arrives,
        # the callback function is called automatically.
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.callback,
            10
        )


    # --------------------------------------------------------------------------
    # CAMERA CALLBACK
    # This function runs automatically every time a new camera frame arrives.
    # It detects all three colours, draws circles around them on the image,
    # and updates the blue_found, green_found, red_found flags for the main loop.
    # --------------------------------------------------------------------------
    def callback(self, data):

        # Convert the ROS image message to an OpenCV image we can work with.
        # We wrap it in a try/except in case the conversion fails.
        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            print(e)
            return

        # Resize to a fixed size so the contour area thresholds are consistent
        image = cv2.resize(image, (320, 240))

        # Convert from BGR to HSV colour space. HSV separates colour (hue) from
        # brightness, which makes colour detection much more reliable than BGR.
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the HSV colour ranges for each colour.
        # Hue goes from 0 to 180 in OpenCV. Green is at 60, blue is at 120.
        # We use sensitivity to widen the range slightly on each side.
        hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])

        hsv_blue_lower = np.array([120 - self.sensitivity, 100, 100])
        hsv_blue_upper = np.array([120 + self.sensitivity, 255, 255])

        # Red is special because it wraps around the hue circle at 0/180.
        # So we need two separate ranges and combine them into one mask.
        hsv_red_lower1 = np.array([0, 100, 100])
        hsv_red_upper1 = np.array([self.sensitivity, 255, 255])
        hsv_red_lower2 = np.array([180 - self.sensitivity, 100, 100])
        hsv_red_upper2 = np.array([180, 255, 255])

        # Create masks — white where the colour is present, black everywhere else
        green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
        blue_mask  = cv2.inRange(hsv_image, hsv_blue_lower,  hsv_blue_upper)
        red_mask1  = cv2.inRange(hsv_image, hsv_red_lower1,  hsv_red_upper1)
        red_mask2  = cv2.inRange(hsv_image, hsv_red_lower2,  hsv_red_upper2)
        red_mask   = cv2.bitwise_or(red_mask1, red_mask2)

        # Combine all three masks so the filtered view shows all detected colours
        rg_mask  = cv2.bitwise_or(red_mask, green_mask)
        all_mask = cv2.bitwise_or(rg_mask, blue_mask)

        # Apply the combined mask to the original image — only detected colour
        # pixels remain visible, everything else turns black
        filtered_img = cv2.bitwise_and(image, image, mask=all_mask)


        # --- GREEN DETECTION ---
        # Find all blobs in the green mask, pick the biggest one,
        # and draw a circle around it on the camera feed image.
        self.green_found = False
        contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Pick the largest contour — this filters out small noise patches
            c = max(contours, key=cv2.contourArea)

            # Only count it if the blob is big enough to be a real object
            if cv2.contourArea(c) > 100:
                self.green_found = True

                # Draw a circle around the detected green blob
                (x, y), radius = cv2.minEnclosingCircle(c)
                cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)

                # Print a terminal message the first time green is detected
                if not self.green_detected:
                    self.get_logger().info('Green object detected')
                    self.green_detected = True


        # --- BLUE DETECTION ---
        # Same as green, but we also record the blob area and horizontal centre
        # position so the robot knows when to stop and which way to steer.
        self.blue_found = False
        self.blue_area  = 0
        contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours_blue) > 0:
            c    = max(contours_blue, key=cv2.contourArea)
            area = cv2.contourArea(c)

            if area > 100:
                self.blue_found = True
                self.blue_area  = area

                # Use moments to find the horizontal centre (centroid) of the blob.
                # m10 / m00 gives the average x position of all pixels in the blob.
                # We use this to steer left or right toward the box.
                M = cv2.moments(c)
                if M['m00'] > 0:
                    self.blue_pos = int(M['m10'] / M['m00'])

                # Draw a circle around the detected blue blob
                (x, y), radius = cv2.minEnclosingCircle(c)
                cv2.circle(image, (int(x), int(y)), int(radius), (255, 0, 0), 2)

                # Record that blue has been seen at least once during this run
                if not self.blue_detected:
                    self.get_logger().info('Blue object detected')
                    self.blue_detected = True


        # --- RED DETECTION ---
        self.red_found = False
        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours_red) > 0:
            c = max(contours_red, key=cv2.contourArea)

            if cv2.contourArea(c) > 100:
                self.red_found = True

                # Draw a circle around the detected red blob
                (x, y), radius = cv2.minEnclosingCircle(c)
                cv2.circle(image, (int(x), int(y)), int(radius), (0, 0, 255), 2)

                # Print a terminal message the first time red is detected
                if not self.red_detected:
                    self.get_logger().info('Red object detected')
                    self.red_detected = True


        # Show the original camera feed with detection circles drawn on it
        cv2.namedWindow('camera_feed', cv2.WINDOW_NORMAL)
        cv2.imshow('camera_feed', image)
        cv2.resizeWindow('camera_feed', 320, 240)

        # Show the filtered feed — only the detected colour pixels are visible
        cv2.namedWindow('filtered_feed', cv2.WINDOW_NORMAL)
        cv2.imshow('filtered_feed', filtered_img)
        cv2.resizeWindow('filtered_feed', 320, 240)

        cv2.waitKey(3)


    # --------------------------------------------------------------------------
    # NAV2 FUNCTIONS
    # These functions handle sending navigation goals to Nav2.
    # Nav2 then plans a path and drives the robot to the target position.
    # This is the same pattern used in lab 4.
    # --------------------------------------------------------------------------

    def send_goal(self, x, y, yaw):
        # Build the goal message with the target position on the map
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Position
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y

        # Orientation — yaw angle converted to quaternion (as in lab 4)
        goal_msg.pose.pose.orientation.z = sin(yaw / 2)
        goal_msg.pose.pose.orientation.w = cos(yaw / 2)

        # Wait until Nav2 is ready to accept goals, then send it
        self.action_client.wait_for_server()
        self.send_goal_future = self.action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)
        self.send_goal_future.add_done_callback(self.goal_response_callback)
        self.goal_in_flight = True

    def goal_response_callback(self, future):
        # This is called when Nav2 either accepts or rejects our goal
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            self.goal_in_flight = False
            return

        # Store the goal handle so we can cancel it later if needed
        self.nav_goal_handle = goal_handle
        self.get_logger().info('Goal accepted')

        # Register another callback for when the robot actually arrives
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        # This is called when the robot has finished navigating to the waypoint.
        # We reset the nav state so the main loop can send the next waypoint.
        self.goal_in_flight  = False
        self.nav_goal_handle = None
        self.get_logger().info('Waypoint reached')

    def feedback_callback(self, feedback_msg):
        # Called repeatedly while the robot is navigating.
        # We do not need feedback right now but the function must exist.
        pass

    def cancel_goal(self):
        # Cancel the current Nav2 goal so we can take over with direct cmd_vel control.
        # This is called when the blue box is spotted mid-navigation.
        # We cancel using the goal handle (stored in goal_response_callback),
        # not the future — this matches the correct Nav2 cancel pattern.
        if self.nav_goal_handle is not None:
            self.nav_goal_handle.cancel_goal_async()
            self.nav_goal_handle = None
        self.goal_in_flight = False


    # --------------------------------------------------------------------------
    # MOVEMENT FUNCTIONS
    # These publish Twist messages directly to /cmd_vel to move the robot.
    # They are used when we want direct control — not during Nav2 navigation.
    # --------------------------------------------------------------------------

    def stop(self):
        # Publish a Twist with all values at zero to stop the robot
        desired_velocity = Twist()
        desired_velocity.linear.x  = 0.0
        desired_velocity.angular.z = 0.0
        self.publisher.publish(desired_velocity)

    def rotate(self):
        # Spin in place by setting only angular velocity.
        # linear.x stays at 0.0 so the robot turns without moving forward.
        # This is used when the robot temporarily loses sight of the blue box.
        desired_velocity = Twist()
        desired_velocity.linear.x  = 0.0
        desired_velocity.angular.z = 0.3
        self.publisher.publish(desired_velocity)

    def approach_blue(self):
        # Drive toward the blue box by moving forward while steering
        # left or right to keep the blue blob centred in the camera.
        desired_velocity = Twist()

        # blue_pos is the horizontal pixel position of the blue blob centre.
        # The image is 320 pixels wide so the centre is at pixel 160.
        # error is how far the blob is from the centre — positive means right, negative means left.
        error = self.blue_pos - 160

        # Set the turning speed proportional to the error.
        # The minus sign makes the robot turn toward the blob:
        # if the blob is to the right (positive error), angular.z goes negative (turn right).
        desired_velocity.angular.z = -float(error) / 160.0 * 0.6

        # Move forward at a slow, controlled speed
        desired_velocity.linear.x = 0.15

        self.publisher.publish(desired_velocity)


# ------------------------------------------------------------------------------
# MAIN FUNCTION
# Sets up the node, starts the ROS spin thread, then runs the state machine loop.
# ------------------------------------------------------------------------------
def main():

    # This function is called when Ctrl+C is pressed.
    # It stops the robot cleanly before shutting down.
    def signal_handler(sig, frame):
        bot.stop()
        rclpy.shutdown()

    rclpy.init(args=None)
    bot = ColourIdentifier()

    signal.signal(signal.SIGINT, signal_handler)

    # rclpy.spin needs to run in a background thread so the camera callback
    # keeps firing while the main loop below is also running.
    # Without this, the main loop would block ROS from receiving messages.
    thread = threading.Thread(target=rclpy.spin, args=(bot,), daemon=True)
    thread.start()

    rate = bot.create_rate(10)

    try:
        while rclpy.ok():

            # ------------------------------------------------------------------
            # STATE: EXPLORING
            # Nav2 is driving the robot through the waypoints one by one.
            # If the blue box is spotted at any point, cancel navigation and
            # switch to approaching immediately.
            # When a waypoint is reached, get_result_callback sets
            # goal_in_flight to False and the next waypoint is sent here.
            # ------------------------------------------------------------------
            if bot.state == 'exploring':

                if bot.blue_found:
                    bot.cancel_goal()
                    bot.stop()
                    bot.state = 'approaching'
                    bot.get_logger().info('Blue box found — approaching')

                elif not bot.goal_in_flight:
                    if bot.waypoint_idx < len(WAYPOINTS):
                        x     = WAYPOINTS[bot.waypoint_idx][0]
                        y     = WAYPOINTS[bot.waypoint_idx][1]
                        theta = WAYPOINTS[bot.waypoint_idx][2]
                        bot.get_logger().info(
                            f'Navigating to waypoint {bot.waypoint_idx}')
                        bot.send_goal(x, y, theta)
                        bot.waypoint_idx += 1

                    else:
                        # All waypoints visited — rotate on the spot to keep scanning
                        bot.rotate()


            # ------------------------------------------------------------------
            # STATE: APPROACHING
            # Nav2 is no longer active. We drive toward the blue box directly
            # using cmd_vel, steering based on where the blob is in the camera.
            # Stop when the blob is large enough that we are about 1 metre away.
            # ------------------------------------------------------------------
            elif bot.state == 'approaching':

                if bot.blue_area >= BLUE_STOP_AREA:
                    # Blob is big enough — we are close enough, stop the robot
                    bot.stop()
                    bot.state = 'stopped'
                    bot.get_logger().info(
                        f'Reached the blue box — stopped (area = {bot.blue_area})')

                elif bot.blue_found:
                    # Blue is visible — drive toward it
                    bot.approach_blue()

                else:
                    # Temporarily lost sight of blue — rotate slowly to find it again
                    bot.rotate()


            # ------------------------------------------------------------------
            # STATE: STOPPED
            # Task is complete. Keep publishing zero velocity to hold position.
            # Print a one-time summary of which colours were detected during the run.
            # ------------------------------------------------------------------
            elif bot.state == 'stopped':
                bot.stop()

                if not bot.summary_printed:
                    bot.get_logger().info('--- Colour detection summary ---')
                    bot.get_logger().info(f'Red detected:   {bot.red_detected}')
                    bot.get_logger().info(f'Green detected: {bot.green_detected}')
                    bot.get_logger().info(f'Blue detected:  {bot.blue_detected}')
                    bot.summary_printed = True


            rate.sleep()

    except ROSInterruptException:
        pass

    # Remember to destroy all image windows before closing node
    cv2.destroyAllWindows()


# Check if the node is executing in the main path
if __name__ == '__main__':
    main()