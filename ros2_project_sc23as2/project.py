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
from math import sin, cos, pi
import signal


# this contour area (in pixels) of the blue blob on the screen when the robot is approximately
# 1m away from the blue box. When the blob reaches this size the robot will stop in the sim world
BLUE_STOP_AREA = 30_000

# list of (x, y, theta) map coordinates the robot will navigate to one by one
# these coordinates were found using Publish Point tool in RViz
WAYPOINTS = [
    (-3.0, -3.0, 0.0),
    (-1.0, -5.0, math.pi),
    (1.65, -11.0, math.pi/2),
]


class Robot(Node):

    def __init__(self):
        super().__init__('robot')

        self.bridge = CvBridge()
        self.sensitivity = 10

        # colour detection flags set to true inside the camera callback whenever
        # each colour is detected in the current frame
        self.red_found = False
        self.green_found = False
        self.blue_found = False

        # overal colour detection flags set to true the first time each colour is 
        # ever detected during the run, stay true even if the colour goes out of view later
        self.green_detected = False
        self.red_detected = False
        self.blue_detected = False

        # for printing final colour summary is only printed once cuz of loop
        self.summary_printed = False

        # pixel area of the blue blob, to decide when to stop approaching blue box
        self.blue_area = 0

        # horizontal pixel position of the centre of the blue blob to steer camera
        # (image is 320 pixels wide so 160 is centre)
        self.blue_pos = 160

        # states (exploring, approaching, stopped)
        self.state = 'exploring'

        # which waypoint to send to nav2 next
        self.waypoint_idx = 0

        # goal_in_flight is true while nav2 is actively navigating to a waypoint
        self.goal_in_flight = False

        # nav_goal_handle stores the handle of the current nav2 goal
        # so we can cancel it mid journey if blue box is spotted
        self.nav_goal_handle = None

        # publisher to send desired_velocity when approaching or rotating
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # action client for nav2
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # subscribe to camera
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.callback,
            10
        )


    def callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            print(e)
            return

        # resize for performance and convert bgr to hsv
        image = cv2.resize(image, (320, 240))
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # colour ranges for green, blue, red (red wraps around hue range so needs two)
        hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])

        hsv_blue_lower = np.array([120 - self.sensitivity, 100, 100])
        hsv_blue_upper = np.array([120 + self.sensitivity, 255, 255])

        hsv_red_lower1 = np.array([0, 100, 100])
        hsv_red_upper1 = np.array([self.sensitivity, 255, 255])
        hsv_red_lower2 = np.array([180 - self.sensitivity, 100, 100])
        hsv_red_upper2 = np.array([180, 255, 255])

        # masks: white for colour, otherwise black 
        # combine masks for red using bitwise_or
        green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
        blue_mask = cv2.inRange(hsv_image, hsv_blue_lower, hsv_blue_upper)
        red_mask1 = cv2.inRange(hsv_image, hsv_red_lower1, hsv_red_upper1)
        red_mask2 = cv2.inRange(hsv_image, hsv_red_lower2, hsv_red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # combine all three
        rg_mask = cv2.bitwise_or(red_mask, green_mask)
        all_mask = cv2.bitwise_or(rg_mask, blue_mask)

        # apply mask to og image, only show pixels where mask is white
        filtered_img = cv2.bitwise_and(image, image, mask=all_mask)

        # green detection and highlight circle
        # get the largest contour to filter noise
        # only count a blob big enough (>100 area)
        self.green_found = False
        contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 100:
                self.green_found = True
                # Draw a circle around the detected green blob
                (x, y), radius = cv2.minEnclosingCircle(c)
                cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                # Print a terminal message the first time green is detected
                if not self.green_detected:
                    self.get_logger().info('Green box spotted')
                    self.green_detected = True

        # blue detection
        # same as green, but also record the blob area and horizontal centre
        # position for steering
        self.blue_found = False
        self.blue_area = 0
        contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours_blue) > 0:
            c = max(contours_blue, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > 100:
                self.blue_found = True
                self.blue_area = area
                # use moments to find the horizontal centre of the blob.
                # m10 / m00 gives the average x position of all pixels in the blob.
                # we use this to steer left or right toward the box.
                M = cv2.moments(c)
                if M['m00'] > 0:
                    self.blue_pos = int(M['m10'] / M['m00'])
                # Draw a circle around the detected blue blob
                (x, y), radius = cv2.minEnclosingCircle(c)
                cv2.circle(image, (int(x), int(y)), int(radius), (255, 0, 0), 2)
                # Record that blue has been seen at least once during this run
                if not self.blue_detected:
                    self.get_logger().info('Blue box spotted')
                    self.blue_detected = True

        # red detection (same as green)
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
                    self.get_logger().info('Red box spotted')
                    self.red_detected = True

        # show both original and filtered camera feeds
        cv2.namedWindow('camera_feed', cv2.WINDOW_NORMAL)
        cv2.imshow('camera_feed', image)
        cv2.resizeWindow('camera_feed', 320, 240)

        cv2.namedWindow('filtered_feed', cv2.WINDOW_NORMAL)
        cv2.imshow('filtered_feed', filtered_img)
        cv2.resizeWindow('filtered_feed', 320, 240)

        cv2.waitKey(3)

    # NAV2 FUNCTIONS
    # for sending goals to nav2, automatic path planning and motion
    def send_goal(self, x, y, yaw):
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y

        goal_msg.pose.pose.orientation.z = sin(yaw / 2)
        goal_msg.pose.pose.orientation.w = cos(yaw / 2)

        self.action_client.wait_for_server()
        self.send_goal_future = self.action_client.send_goal_async(goal_msg, 
                                                                   feedback_callback=self.feedback_callback)
        self.send_goal_future.add_done_callback(self.goal_response_callback)
        self.goal_in_flight = True


    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal waypoint rejected')
            self.goal_in_flight = False
            return
        # store the goal handle so we can cancel it later if needed
        self.nav_goal_handle = goal_handle
        self.get_logger().info('Goal waypoint accepted')
        # register another callback for when the robot actually arrives
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        # called when the robot has finished navigating to the waypoint
        # reset the nav state so the main loop can send the next waypoint
        self.goal_in_flight = False
        self.nav_goal_handle = None
        self.get_logger().info('Waypoint reached')

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback

    def cancel_goal(self):
        # cancel the current nav2 goal so we can take over with direct cmd_vel control.
        # called if blue box is spotted
        if self.nav_goal_handle is not None:
            self.nav_goal_handle.cancel_goal_async()
            self.nav_goal_handle = None
        self.goal_in_flight = False

    # MOVEMENT FUNCTIONS
    # to publish twist messages to /cmd_vel
    def stop(self):
        desired_velocity = Twist()
        desired_velocity.linear.x = 0.0
        desired_velocity.angular.z = 0.0
        self.publisher.publish(desired_velocity)

    def approach_blue(self):
        # drive toward the blue box by moving forward while steering left or 
        # right to keep the blue blob centred in the camera
        desired_velocity = Twist()

        # blue_pos is the horizontal ppx position of blob centre (320px image so at 160)
        # error is how far the blob is from centre, +ve for right, -ve for left
        error = self.blue_pos - 160

        # turning speed proportional to error (minus sign makes the robot turn toward the blob)
        # if the blob is to the right (positive error), angular.z goes negative (turn right)
        desired_velocity.angular.z = -float(error) / 160.0 * 0.6
        desired_velocity.linear.x = 0.15

        self.publisher.publish(desired_velocity)


def main():

    def signal_handler(sig, frame):
        bot.stop()
        rclpy.shutdown()

    rclpy.init(args=None)
    bot = Robot()

    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(bot,), daemon=True)
    thread.start()

    rate = bot.create_rate(10)

    try:
        while rclpy.ok():

            # EXPLORING STATE
            if bot.state == 'exploring':
                # nav2 drives robot through the waypoints one by one, cancel nav2 if blue bpx
                # spotted and switch to approaching
                if bot.blue_found:
                    bot.cancel_goal()
                    bot.stop()
                    bot.state = 'approaching'
                    bot.get_logger().info("Blue box spotted: approaching now")
                # when a waypoint is reached, get_result_callback sets goal_in_flight to False and 
                # the next waypoint is sent here
                elif not bot.goal_in_flight:
                    if bot.waypoint_idx < len(WAYPOINTS):
                        x = WAYPOINTS[bot.waypoint_idx][0]
                        y = WAYPOINTS[bot.waypoint_idx][1]
                        theta = WAYPOINTS[bot.waypoint_idx][2]
                        bot.get_logger().info(
                            f'Navigating to waypoint {bot.waypoint_idx + 1} of {len(WAYPOINTS)}: x={x}, y={y}, theta={theta}')
                        bot.send_goal(x, y, theta)
                        bot.waypoint_idx += 1
                    else:
                        # all waypoints done and blue not yet found, stop
                        bot.stop()
                        bot.get_logger().info("All waypoints visited, blue box not found")

            # APPROACHING STATE
            # drive toward the blue box directly using cmd_vel, steering based on where the blob is 
            # in the camera, stop when the blob is large enough that we are about 1 metre away
            elif bot.state == 'approaching':
                if bot.blue_area >= BLUE_STOP_AREA:
                    bot.stop()
                    bot.state = 'stopped'
                    bot.get_logger().info(
                        f"Blue box has been reached (area = {bot.blue_area})")
                elif bot.blue_found:
                    bot.approach_blue()
                else:
                    bot.stop()
                    bot.get_logger().info('Lost blue box, waiting')

            # STOPPED STATE
            # task complete, stop bot
            # print final summary of which colours were detected during the run
            elif bot.state == 'stopped':
                bot.stop()
                if not bot.summary_printed:
                    bot.get_logger().info('--- !!! Colour detection summary !!! ---')
                    bot.get_logger().info(f'Red detected: {bot.red_detected}')
                    bot.get_logger().info(f'Green detected: {bot.green_detected}')
                    bot.get_logger().info(f'Blue detected: {bot.blue_detected}')
                    bot.summary_printed = True

            rate.sleep()

    except ROSInterruptException:
        pass

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()