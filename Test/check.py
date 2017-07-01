import math

(p_c_x, p_c_y) = (100, 100)
(c_x, c_y) = (50, 110)
length = math.sqrt(math.fabs(p_c_x - c_x)**2 + math.fabs(p_c_y - c_y)**2)
angle = math.acos((c_x - p_c_x)/length)
print(length)
print(angle)
print(math.degrees(angle))