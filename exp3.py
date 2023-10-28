import cv2
interface = cv2.imread("interface/as2c.jpeg")
interface = cv2.resize(interface, (1240, 680))
bl = cv2.imread("interface/blackpage.png")
bl = cv2.resize(bl, (1230 - 800, 670 - 10))
interface[10:670, 800:1230] = bl
interface = cv2.rectangle(interface, (127, 116), (640 + 127, 116 + 486), (0, 255, 50), 3)
cv2.imwrite("newinterface.jpg",interface)