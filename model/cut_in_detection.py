import cv2
from ultralytics import YOLO
import numpy as np
import statistics
import os

# Place this file and best.pt file in idd_mm_primary folder

model = YOLO("best.pt")

def find_distance(height_px):
	sensor_height = 3.02
	focal_length = 2.12
	frame_height = 1088
	distance = (focal_length*1600*frame_height)/(height_px*sensor_height)

	return distance/1000

drive_sequence = "d0" # d0, d1, d2
files = os.listdir(f'./idd_multimodal/primary/{drive_sequence}/leftCamImgs')

distance = dict()
vel_angle = dict()
warning = 0
count = 0
# Use slicing files[:] to loop through small batch of files
for file in files[3500:]:
	frame = cv2.imread(f'./idd_multimodal/primary/{drive_sequence}/leftCamImgs/{file}')

	results = model.track(frame, conf=0.75, persist=True)
	annotated_frame = results[0].plot()

	for r in results:
		bounding = r.boxes.xywh # Coordinates of bounding boxes # Tensor
		box = r.boxes.id # Tensor

		# If no object is detected clear up distance and vel_angle
		if box == None:
			distance = dict()
			vel_angle = dict()
			continue

		temp = list(distance.keys())

		for key in temp:
			if key not in box:
				distance.pop(key)

		for j in range(bounding.shape[0]):
			x, y, w, h = bounding[j,:]

			# Initialize distance of object
			if int(box[j]) not in distance.keys():
				distance[int(box[j])] = [0, 0]
				distance[int(box[j])][1] = float(find_distance(h))
				continue
			else:
				distance[int(box[j])][0] = distance[int(box[j])][1]
				distance[int(box[j])][1] = float(find_distance(h))

				v = abs(distance[int(box[j])][1]-distance[int(box[j])][0])/0.033


				if x+w/2<960: # Find difference between consecutive angles to determine if a car is cutting in
					cv2.line(annotated_frame, (0, 1080), ((int(x)+int(w/2)), (int(y)-int(h/2))), (0, 255, 0))
					cv2.line(annotated_frame, (0, 1080), (960, 480), (0, 0, 255))
					m1 = (y-h/2-1080)/(x+w/2-0)
					angle = float((np.arctan(abs(m1))*180/3.14)-(np.arctan(abs(-5/8))*180/3.14))
				else:
					cv2.line(annotated_frame, (1920, 1080), ((int(x)-int(w/2)), (int(y)-int(h/2))), (0, 255, 0))
					cv2.line(annotated_frame, (1920, 1080), (960, 480), (0, 0, 255))
					m1 = (y-h/2-1080)/(x+w/2-1920)
					angle = float((np.arctan(abs(m1))*180/3.14)-(np.arctan(abs((5/8)))*180/3.14))

				# Initialize angle with the reference line
				if int(box[j]) not in vel_angle.keys():
					vel_angle[int(box[j])] = list()
					vel_angle[int(box[j])].append([v])
					vel_angle[int(box[j])].append([angle])

				else:
					# Using previous 6 frames to determine ideal velocity and angle_diff values
					if len(vel_angle[int(box[j])][0]) == 6:
						vel = statistics.median(vel_angle[int(box[j])][0])
						ttc = distance[int(box[j])][1]/vel

						angle_diff = vel_angle[int(box[j])][1][-1]-vel_angle[int(box[j])][1][0]

						vel_angle[int(box[j])][0].pop(0) # Remove the first stored value in the velocity array, i.e. the value at index 0
						vel_angle[int(box[j])][1].pop(0) # Remove the first stored value in the angle array, i.e. the value at index 0

						if ttc<0.7 and angle_diff<0 and angle<6:
							print(int(box[j]), ttc)
							# Set warning to 1 for next 20 frames
							warning = 1

							

					vel_angle[int(box[j])][0].append(v)
					vel_angle[int(box[j])][1].append(angle)


	# Print the cut-in warning for next 20 frames
	if warning == 1 and count < 20:
		cv2.putText(annotated_frame, "Cut-In", (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
		count += 1
	else:
		count = 0
		warning = 0

	cv2.imshow("YOLOv8 Tracking", annotated_frame)

	# Press q to exit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		exit()
