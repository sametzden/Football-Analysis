from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import pickle
import os
import sys
import cv2
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)  # Initialize the YOLO model
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=120,  # 4s at 30fps — keeps lost tracks alive longer before assigning a new ID
            minimum_matching_threshold=0.6  # lowered from 0.8 → re-matches occluded/fast players more reliably
        )

    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
            


    def interpolate_ball_position(self, ball_positions):
        ball_positions = [x.get(1,{}).get("bbox", []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])

        #interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions


    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size],conf =0.1)
            detections += detections_batch       
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks


        detections = self.detect_frames(frames)

        tracks = {
            "players": [],      # list of dicts per frame: {track_id: {"bbox": ...}}
            "goalkeepers": [],  # same structure — kept separate for team assignment
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            if detection.boxes is not None:
                cls_names = detection.names
                cls_names_inv = {v: k for k, v in cls_names.items()}

                # Convert to supervision detection format
                detection_supervision = sv.Detections.from_ultralytics(detection)

                # NOTE: goalkeepers are NOT converted to players anymore.
                # They are tracked separately so we can assign them to a team
                # using their field position (left/right half of the pitch).

                detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
                tracks["players"].append({})
                tracks["goalkeepers"].append({})
                tracks["referees"].append({})
                tracks["ball"].append({})

                for frame_detection in detection_with_tracks:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    track_id = frame_detection[4]

                    if cls_id == cls_names_inv.get("player"):
                        tracks["players"][frame_num][track_id] = {"bbox": bbox}

                    if cls_id == cls_names_inv.get("goalkeeper"):
                        tracks["goalkeepers"][frame_num][track_id] = {"bbox": bbox}

                    if cls_id == cls_names_inv.get("referee"):
                        tracks["referees"][frame_num][track_id] = {"bbox": bbox}

                for frame_detection in detection_supervision:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]

                    if cls_id == cls_names_inv.get("ball"):
                        tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(int(x_center),int(y2)),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame
    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame
   
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # draw a semi-transparent rectangle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350,850), (1900, 970), (255,255,255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        #get the number of times each team had ball control till current frame
        team1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]

        team1 = team1_num_frames/(team1_num_frames+team2_num_frames) 
        team2 = team2_num_frames/(team1_num_frames+team2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team1:.1%}", (1370,900), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        cv2.putText(frame, f"Team 2 Ball Control: {team2:.1%}", (1370,940), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_vidoe_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            goalkeeper_dict = tracks["goalkeepers"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw player tracks
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_traingle(frame, player["bbox"], (255, 0, 0))

            # Draw Goalkeepers — same ellipse but with a distinct white outline to distinguish them
            for track_id, goalkeeper in goalkeeper_dict.items():
                color = goalkeeper.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, goalkeeper["bbox"], color, track_id)
                # Extra white ring to visually distinguish goalkeeper from outfield players
                frame = self.draw_ellipse(frame, goalkeeper["bbox"], (255, 255, 255))

                if goalkeeper.get("has_ball", False):
                    frame = self.draw_traingle(frame, goalkeeper["bbox"], (255, 0, 0))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw Ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            # Add team ball control annotation
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            output_vidoe_frames.append(frame)

        return output_vidoe_frames

    def merge_fragmented_tracks(self, tracks, max_frame_gap=60, max_distance=150):
        """
        Merges broken tracklets of the same person across occlusions.
        A tracklet is defined by its start/end frame, position, and assigned team.
        """
        import math
        
        for category in ["players", "goalkeepers", "referees"]:
            track_history = {} # track_id -> {start_frame, end_frame, start_pos, end_pos, team}
            
            # Step 1: Collect track history
            for frame_num, frame_tracks in enumerate(tracks[category]):
                for track_id, track_data in frame_tracks.items():
                    pos = track_data.get("position", [0, 0])
                    team = track_data.get("team")
                    
                    if track_id not in track_history:
                        track_history[track_id] = {
                            "start_frame": frame_num,
                            "end_frame": frame_num,
                            "start_pos": pos,
                            "end_pos": pos,
                            "team": team
                        }
                    else:
                        track_history[track_id]["end_frame"] = frame_num
                        track_history[track_id]["end_pos"] = pos
                        if track_history[track_id]["team"] is None and team is not None:
                            track_history[track_id]["team"] = team

            # Step 2: Greedly merge tracks
            sorted_tracks = sorted(track_history.keys(), key=lambda t: track_history[t]["start_frame"])
            id_mapping = {} # old_id -> new_id
            
            for i, current_id in enumerate(sorted_tracks):
                current_data = track_history[current_id]
                best_match_id = None
                best_match_distance = float('inf')
                
                for j in range(i - 1, -1, -1):
                    prev_id = sorted_tracks[j]
                    
                    # Follow mapping to the root ID
                    root_prev_id = prev_id
                    while root_prev_id in id_mapping:
                        root_prev_id = id_mapping[root_prev_id]
                        
                    prev_data = track_history[root_prev_id]
                    frame_gap = current_data["start_frame"] - prev_data["end_frame"]
                    
                    if 0 < frame_gap <= max_frame_gap:
                        dist = math.hypot(
                            current_data["start_pos"][0] - prev_data["end_pos"][0],
                            current_data["start_pos"][1] - prev_data["end_pos"][1]
                        )
                        if dist <= max_distance and dist < best_match_distance:
                            # Strict match for Team (if known)
                            if current_data["team"] is not None and prev_data["team"] is not None:
                                if current_data["team"] == prev_data["team"]:
                                    best_match_id = root_prev_id
                                    best_match_distance = dist
                            else:
                                best_match_id = root_prev_id
                                best_match_distance = dist
                
                if best_match_id is not None:
                    id_mapping[current_id] = best_match_id
                    track_history[best_match_id]["end_frame"] = current_data["end_frame"]
                    track_history[best_match_id]["end_pos"] = current_data["end_pos"]

            # Step 3: Apply the id_mapping to the actual tracks dictionary
            for frame_num, frame_tracks in enumerate(tracks[category]):
                new_frame_tracks = {}
                for track_id, track_data in frame_tracks.items():
                    root_id = track_id
                    while root_id in id_mapping:
                        root_id = id_mapping[root_id]
                    new_frame_tracks[root_id] = track_data
                tracks[category][frame_num] = new_frame_tracks


    def interpolate_missing_frames(self, tracks):
        """
        After merging tracks, fills in the missing gap frames linearly to prevent bounding box teleportation.
        """
        for category in ["players", "goalkeepers", "referees"]:
            track_dict = {} 
            for frame_num, frame_tracks in enumerate(tracks[category]):
                for track_id, track_data in frame_tracks.items():
                    if track_id not in track_dict:
                        track_dict[track_id] = {}
                    track_dict[track_id][frame_num] = track_data
            
            for track_id, frames_data in track_dict.items():
                sorted_frames = sorted(frames_data.keys())
                if not sorted_frames:
                    continue
                
                start_frame = sorted_frames[0]
                end_frame = sorted_frames[-1]
                
                for f in range(start_frame + 1, end_frame):
                    if f not in frames_data:
                        prev_f = max(k for k in sorted_frames if k < f)
                        next_f = min(k for k in sorted_frames if k > f)
                        
                        prev_data = frames_data[prev_f]
                        next_data = frames_data[next_f]
                        ratio = (f - prev_f) / (next_f - prev_f)
                        
                        interpolated_data = {}
                        
                        def interpolate_array(arr1, arr2):
                            if arr1 is None or arr2 is None: return arr1
                            return [a + (b - a) * ratio for a, b in zip(arr1, arr2)]
                            
                        # Standard fields
                        interpolated_data["bbox"] = interpolate_array(prev_data.get("bbox"), next_data.get("bbox"))
                        interpolated_data["position"] = interpolate_array(prev_data.get("position"), next_data.get("position"))
                        
                        # Transformed positions
                        if "position_transformed" in prev_data and "position_transformed" in next_data:
                            interpolated_data["position_transformed"] = interpolate_array(prev_data["position_transformed"], next_data["position_transformed"])
                        
                        # Adjusted positions
                        if "position_adjusted" in prev_data and "position_adjusted" in next_data:
                            interpolated_data["position_adjusted"] = interpolate_array(prev_data["position_adjusted"], next_data["position_adjusted"])
                        
                        # Copy static metadata
                        for key in ["team", "team_color", "has_ball"]:
                            if key in prev_data: 
                                interpolated_data[key] = prev_data[key]
                        
                        tracks[category][f][track_id] = interpolated_data
