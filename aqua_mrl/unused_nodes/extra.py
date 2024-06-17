# M = cv2.moments(pred)
        # rect = cv2.minAreaRect(pred)
        # try:
        #     rect = cv2.minAreaRect(pred)
        #     box = cv2.boxPoints(rect)
        #     box = np.int0(box)
        #     cv2.drawContours(pred,[box],0,(0,0,255),2)
        #     cX = int(M["m10"] / M["m00"])
        #     cY = int(M["m01"] / M["m00"])
        #     cv2.circle(pred, (cX, cY), 2, (0, 255, 0), -1)
        # except:
        #     print('MISSED')
        # self.centroid_x = int(M["m10"] / M["m00"])
        # edge_map = cv2.Canny(pred,50,200)
        # zeros = np.zeros((300,400))
        # lines = cv2.HoughLinesP(edge_map, 1, np.pi / 180, 20, None, 30, 10)
        # entered = False
        # if lines is not None:
        #     line_combos = list(combinations(lines, 2))
        #     line_votes = np.zeros(len(line_combos))
        #     for i, lp in enumerate(list(combinations(lines, 2))):
        #         l1 = lp[0][0]
        #         l2 = lp[1][0]
        #         if l1[2] == l1[0]: 
        #             theta1 = np.arctan(np.inf)
        #         else:
        #             theta1 = np.arctan(np.divide(l1[3] - l1[1], l1[2] - l1[0]))

        #         if l2[2] == l2[0]:
        #             theta2 = np.arctan(np.inf)
        #         else:
        #             theta2 = np.arctan(np.divide(l2[3] - l2[1], l2[2] - l2[0]))
            
        #         angle = np.abs(theta1 - theta2)
        #         angle = min(angle, np.abs(np.pi - angle))
        #         if angle < 0.34:
        #             l1_midpoint = (l1[0] + l1[2])/2, (l1[1] + l1[3])/2
        #             l2_midpoint = (l2[0] + l2[2])/2, (l2[1] + l2[3])/2
        #             if np.abs(l1_midpoint[0] - l2_midpoint[0]) > 20 and np.abs(l1_midpoint[1] - l2_midpoint[1]) > 20:
        #                 l1_length = np.sqrt(np.square(l1[3] - l1[1]) + np.square(l1[2] - l1[0]))
        #                 l2_length = np.sqrt(np.square(l2[3] - l2[1]) + np.square(l2[2] - l2[0]))
        #                 line_votes[i] = l1_length + l2_length
        #                 entered = True
        #             # cv2.line(zeros, (l1[0], l1[1]), (l1[2], l1[3]), 255, 1, cv2.LINE_AA)
        #             # cv2.line(zeros, (l2[0], l2[1]), (l2[2], l2[3]), 255, 1, cv2.LINE_AA)
        #             # break
        #     # if not entered:
        #     #     self.missed += 1
        #     # print('MISSED: ', self.missed)
        #     if entered: 
        #         pipeline = line_combos[np.argmax(line_votes)]
        #         p1 = pipeline[0][0]
        #         p2 = pipeline[1][0]
        #         if p1[0] == p1[2]:
        #             start = (int(p1[0]),0)
        #             end = (int(p1[0]),300)
        #         else:
        #             slope = np.divide(p1[3] - p1[1], p1[2] - p1[0])
        #             b = p1[1] - slope * p1[0]
        #             start = (int(-b/slope), 0)
        #             end = (int((300 - b)/slope), 300)
        #         cv2.line(zeros, start, end, 255, 1 , cv2.LINE_AA)
        #         if p2[0] == p2[2]:
        #             start = (p2[0],0)
        #             end = (p2[0],300)
        #         else:
        #             slope = np.divide(p2[3] - p2[1], p2[2] - p2[0])
        #             b = p2[1] - slope * p2[0]
        #             start = (int(-b/slope), 0)
        #             end = (int((300 - b)/slope), 300)
        #         cv2.line(zeros, start, end, 255, 1 , cv2.LINE_AA)
        #     else:
        #         self.missed += 1
        #         print('MISSED: ', self.missed)