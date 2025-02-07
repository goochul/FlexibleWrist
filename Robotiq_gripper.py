from pyRobotiqGripper import RobotiqGripper

gripper = RobotiqGripper()

gripper.activate()
gripper.calibrate(0, 40)

gripper.open()
gripper.close()
gripper.goTo(5)
position_in_bit = gripper.getPosition()
print(position_in_bit)

gripper.goTomm(25)
position_in_mm = gripper.getPositionmm()
print(position_in_mm)

gripper.printInfo()
