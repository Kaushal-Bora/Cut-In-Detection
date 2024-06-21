import cv2
from ultralytics import YOLO
import numpy as np
import statistics
import os
import numpy as np
# Place this file and best.pt file in idd_mm_primary folder

model = YOLO("best.onnx")

def find_distance(height_px, vehicle_class):
	real_height = {0: 1000, 1:1600, 2:4000} # In mm
	sensor_height = 3.02
	focal_length = 2.12
	frame_height = 1080
	distance = (focal_length*real_height[int(vehicle_class)]*frame_height)/(height_px*sensor_height)
	return distance/1000

drive_sequence = "d1" # d0, d1, d2
files = os.listdir(f'./idd_multimodal/primary/{drive_sequence}/leftCamImgs')


# Frame dimensions and lane markings
frame_width = 1920
frame_height = 1080
center_frame_w = frame_width/2 - 80
center_frame_h = frame_height/2
left_lane_p1 = (int(center_frame_w - frame_width*0.33), frame_height)
left_lane_p2 = (int(center_frame_w - frame_width*0.02), int(frame_height*0.5))
right_lane_p1 = (int(center_frame_w + frame_width*0.33), frame_height)
right_lane_p2 = (int(center_frame_w + frame_width*0.02), int(frame_height*0.5))
slope_l1 = abs((left_lane_p2[1]-left_lane_p1[1])/(left_lane_p2[0]-left_lane_p1[0]))
slope_l2 = abs((right_lane_p2[1]-right_lane_p1[1])/(right_lane_p2[0]-right_lane_p1[0]))

distance = dict()
vel_angle = dict()
warning = 0
count = 0
# Use slicing files[:] to loop through small batch of files
for file in files[3500:]:
	frame = cv2.imread(f'./idd_multimodal/primary/{drive_sequence}/leftCamImgs/{file}')
	try:
		results = model.track(frame, conf=0.75, persist=True, iou=0.95, imgsz=(1920,1088))
	except np.linalg.LinAlgError:
		continue
	annotated_frame = results[0].plot()

	for r in results:
		bounding = r.boxes.xywh # Coordinates of bounding boxes # Tensor
		box = r.boxes.id # Tensor
		vehicle_class = r.boxes.cls

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
				distance[int(box[j])][1] = float(find_distance(h, vehicle_class[j]))-1.5
				continue
			else:
				distance[int(box[j])][0] = distance[int(box[j])][1]
				distance[int(box[j])][1] = float(find_distance(h, vehicle_class[j]))-1.5

				v = abs(distance[int(box[j])][1]-distance[int(box[j])][0])*15 # Time interval is 1/15, so distance/time_interval


				if x+w/2<center_frame_w: # Find difference between consecutive angles to determine if a car is cutting in
					cv2.line(annotated_frame, left_lane_p1, ((int(x)+int(w/2)), (int(y)-int(h/2))), (0, 255, 0))
					cv2.line(annotated_frame, left_lane_p1, left_lane_p2, (255, 255, 255), 2)
					m = (y-(h/2)-left_lane_p1[1])/(x+(w/2)-left_lane_p1[0])
					angle = float(((np.arctan(abs(m))-(np.arctan(slope_l1)))*180/np.pi))
				else:
					cv2.line(annotated_frame, right_lane_p1, ((int(x)-int(w/2)), (int(y)-int(h/2))), (0, 255, 0))
					cv2.line(annotated_frame, right_lane_p1, right_lane_p2, (255, 255, 255), 2)
					m = (y-(h/2)-right_lane_p1[1])/(x-(w/2)-right_lane_p1[0])
					angle = float(((np.arctan(abs(m))-(np.arctan(slope_l2)))*180/np.pi))

				# Initialize angle with the reference line
				if int(box[j]) not in vel_angle.keys():
					vel_angle[int(box[j])] = list()
					vel_angle[int(box[j])].append([v])
					vel_angle[int(box[j])].append([angle])

				else:
					# Using previous 6 frames to determine ideal velocity and angle_diff values
					if len(vel_angle[int(box[j])][0]) == 5:
						vel = statistics.median(vel_angle[int(box[j])][0])
						angle_stdev = statistics.stdev(vel_angle[int(box[j])][1])
						ttc = distance[int(box[j])][1]/vel
						
						angle_diff = vel_angle[int(box[j])][1][-1]-vel_angle[int(box[j])][1][-2]
						print("stddev", int(box[j]), ttc, angle_diff, statistics.stdev(vel_angle[int(box[j])][1]), angle)

						vel_angle[int(box[j])][0].pop(0) # Remove the first stored value in the velocity array, i.e. the value at index 0
						vel_angle[int(box[j])][1].pop(0) # Remove the first stored value in the angle array, i.e. the value at index 0

						# if ttc<0.7 and angle_diff<0 and angle<5:
						if ttc<0.8 and angle_diff<0 and angle_stdev > 1.5 and angle<5:
							print(int(box[j]), ttc)
							# Set warning to 1 for next 20 
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
