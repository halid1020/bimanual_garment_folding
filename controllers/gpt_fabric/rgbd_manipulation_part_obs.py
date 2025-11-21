import cv2 as cv
import numpy as np
import os
import re
import requests
import time
from .manipulation import RGB_manipulation, encode_image
from agent_arena.utilities.save_utils import save_mask, save_colour
from huggingface_hub import login
HF_TOKEN = os.environ["HF_TOKEN"]
login(token=HF_TOKEN)
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch

class RGBD_manipulation_part_obs(RGB_manipulation):
    """
    This is the manipulation class for the RGBD observation input (paper's method).
    Added attributes:
        re_consider: whether to enable the evaluation module
        in_context_learning: whether to use in-context learning
        demo_dir: the directory of the demonstration data
    
    """
    
    def __init__(self, re_consider=True,in_context_learning=False,demo_dir="./tmp/Manual_test14"):
        
        super().__init__()
        self.re_consider=re_consider
        self.in_context_learning=in_context_learning
        self.demo_dir=demo_dir
        

        model_id = "google/gemma-3-4b-it"

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="cuda:0",
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_id)

        
        
    def save_obs(self, image, rgbd=None, specifier="init"):
        """
        Save the observation to the specified directory in the formate of image and rgbd(.npy).
        Input:
            image: the image to be saved
            rgbd: the rgbd to be saved (default to be None)
            specifier: the specifier (usually should be the number of step) of the observation.
        Output:
            img_path: the path of the saved image
        """
        
        save_name_image = osp.join(self.obs_dir, "image")
        save_image = Image.fromarray(image)
        img_path=save_name_image+'_'+specifier+'.png'
        save_image.save(img_path)
        print('observation save to {} \n'.format(img_path))
        
        if rgbd is not None:
            save_name_rgbd=osp.join(self.obs_dir,"RGBD")
            rgbd_path=save_name_rgbd+'_'+specifier+'.npy'
            np.save(rgbd_path, rgbd)
        return img_path
       


    
    def get_center_point_bounding_box(self,rgb,mask,need_box=False, goal_config=False):
        """
        Get the center point of the bounding box of the fabric in the image and save the image with the center point marked.
        
        Input:
            img_path: the path of the image
            depth: the depth image of the fabric
            need_box: whether to draw the bounding box in the image
        Output:
            center_point_pixel: the pixel of the center point of the bounding box
            rgb: the image with the center point marked
        
        """
        
        
        top,bottom,left,right=self.get_bounds(mask)

        
        center_point_pixel=[((right-left)//2)+left,((top-bottom)//2)+bottom]        
        cv.circle(rgb,(center_point_pixel[0],center_point_pixel[1]),4,(0,0,0),-1)  
        
        
        if need_box:
            for i in range(bottom,top,4):
                rgb[i][left]=[255,0,0]
                rgb[i][right]=[255,0,0]

            for j in range(left,right,4):
                rgb[bottom][j]=[255,0,0]
                rgb[top][j]=[255,0,0]
                
                
        if goal_config:
            goal_top=center_point_pixel[0]+(self.goal_height)//2
            goal_bottom=center_point_pixel[0]-(self.goal_height)//2
            goal_left=center_point_pixel[1]-(self.goal_height)//2
            goal_right=center_point_pixel[1]+(self.goal_height)//2
        ## Draw the flattened box
            for i in range(goal_bottom,goal_top,2):
                rgb[i][goal_left]=[255,255,255]
                rgb[i][goal_right]=[255,255,255]
                
            for j in range(goal_left,goal_right,2):
                rgb[goal_bottom][j]=[255,255,255]
                rgb[goal_top][j]=[255,255,255]
        
        
        
        
        # save_image = Image.fromarray(rgb)
        # save_image.save(osp.join(save_obs_dir,'processed.png'))
        # np.save("init_obs.npy",obs)
        return center_point_pixel,rgb   


    

    def response_process(self,response_message,messages=None):
        """
        Process the response from GPT to get the picking point, direction, distance. Map the picking pixel to 3D coordinate
        and then use move direction and distance to calculate the placing point.
        
        return both picking point and placing point in 3D coordinate and pixel coordinate.
        
        If the response doesn't contain pick point, direction, distance, return None to let the recal module to ask GPT again.
        
        Input:
            response: the response from GPT
        Output:
            pick_coords: the 3D coordinate of the picking point
            place_coords: the 3D coordinate of the placing point
            pick_pixel: the pixel coordinate of the picking point
            place_pixel: the pixel coordinate of the placing point
        
        """
        
        
        # Use regular expression to extract the pick point, direction and distance.
        pick_pattern = r'Pick point:.*?\[(.*?)\]'
        direction_pattern=r'Moving direction:.*?(\d+/\d+)'
        distance_pattern=r'Moving distance:.*?(\d+\.?\d*)'

        pick_match = re.search(pick_pattern, response_message)
        direction_match = re.search(direction_pattern, response_message)
        distance_match = re.search(distance_pattern, response_message)

        # Get pick point (pixel) from GPT response and transform it to 3D coordinate.
        if not pick_match:
            return None,None,None,None
        pick_coords = [int(val) for val in pick_match.group(1).split(',')]
        pick_pixel=pick_coords
        
        pick_coords=camera_utils.find_nearest(self.pixel_coords,pick_coords[1],pick_coords[0])# map the pixel to 3D coordinate
        
        pick_coords=self.pixel_coords[pick_coords[0]][pick_coords[1]] # The 3D coordinate of the picking point
        
            
        # Get moving direction and distance from GPT response.
        moving_direction = direction_match.group(1) if direction_match else None
        if moving_direction is None:
            return None,None,None,None

        numerator, denominator = moving_direction.split('/')
        moving_direction=float(numerator)/float(denominator)
        
        
        moving_distance = float(distance_match.group(1)) if distance_match else None
        if moving_distance is None:
            return None,None,None,None

        # Calculate the placing point based on the picking point, moving direction and distance.
        curr_config=self.env.get_current_config()
        dimx,dimy=curr_config['ClothSize']
        size=max(dimx,dimy)*self.env.cloth_particle_radius

        actual_direction=moving_direction*np.pi
        actual_distance=moving_distance*size

        delta_x=actual_distance*np.sin(actual_direction)
        delta_y=actual_distance*np.cos(actual_direction)


        place_coords = pick_coords.copy()
        place_coords[0]+=delta_x
        place_coords[2]+=delta_y
        
        
        # calculate the pixel coordinate of the placing point
        pixel_size=max(self.goal_height,self.goal_width)
        delta_x_pixel=int(pixel_size*np.cos(actual_direction)*moving_distance)
        delta_y_pixel=int(pixel_size*np.sin(actual_direction)*moving_distance)
        
        place_pixel=[pick_pixel[0]+delta_x_pixel,pick_pixel[1]-delta_y_pixel]

            
        return pick_pixel,place_pixel
    
    
    
    def vis_result(self,place_pixel,pick_pixel=None,img_path=None,img=None):
        """
        Visualize the result of the pick-and-place action.
        If provide both place pixel and pick pixel, draw a circle at the pick pixel and an arrow pointing to the place pixel.
        If only provide place pixel, draw a circle at the place pixel (This is to visualize the action from last step).
        
        Input:
            place_pixel: the pixel coordinate of the placing point
            pick_pixel: the pixel coordinate of the picking point
            img_path: the path of the image (If no image is provided, use the img_path to load the image)
            img: the image to be visualized
        Output:
            img: the image with the pick-and-place action visualized
        """
        
        if img_path:
            img=cv.imread(img_path)
        if pick_pixel is not None:
            cv.circle(img, (int(pick_pixel[0]), int(pick_pixel[1])), 5, (0, 255, 0), 2)
            cv.arrowedLine(img, (int(pick_pixel[0]), int(pick_pixel[1])), (int(place_pixel[0]), int(place_pixel[1])), (0, 255, 0), 2)
            cv.circle(img,(int(place_pixel[0]), int(place_pixel[1])), 5, (128, 0, 128), 2)
        else:
            cv.circle(img, (int(place_pixel[0]), int(place_pixel[1])), 3, (0, 0, 255), 2)
        return img



    def _cal_direction(self,start,end):
        """
        Given a start point and end point, calculate the direction of the vector from start to end.
        
        Input:
            start: the start point
            end: the end point
        Output:
            angle: the angle of the vector from start to end
        """
        vector=[end[0]-start[0],start[1]-end[1]]
                
        angle=np.arctan2(vector[1],vector[0])
        angle=angle/np.pi
        if angle<0.125:
            angle+=2
        
        return angle



                
    def recal(self,response_message,place_pixel,pick_pixel,center,img,depth_img=None,last_pick_point=None, last_pick_point_oppo=None):
        
        """
        This is the module to check the correctness of the predicted pick-and-place action (recal module or Evaluation module in the paper).
        It will check whether the picking point is close to last picking point and whether the move direction approximately aligns with the 
        direction starting from the center point to the chosen picking point. 
        
        It will return the check results of both direction check and picking point appoximity check with the visualization of the action.
        
        Input:
            response_message: the response message from GPT
            place_pixel: the pixel coordinate of the placing point
            pick_pixel: the pixel coordinate of the picking point
            center: the pixel coordinate of the center point
            img: the image to be visualized
            depth_img: the depth image of the fabric
            last_pick_point: the pixel coordinate of the last picking point
            last_pick_point_oppo: the pixel coordinate of the symmetric point of the last picking point
        
        Output:
            correct_message: the correction message to be sent to the GPT for re-consideration
            check_result: whether the picking point is close to the last picking point and the move direction is approximately correct
            direction_check: whether the move direction is approximately correct

        """
        # 0. setup the parameters
        
        # Whether to check the directions only.  
        # **This is for ablation study, normally should be false**
        check_directions_only=False
        
        # Visualization of the predicted action
        img=self.vis_result(place_pixel=place_pixel,pick_pixel=pick_pixel,img=img.copy())        
        vis_result_path=self.paths['processed vis image']
        encoded_vis_result=encode_image(vis_result_path)
        
        
        if depth_img is not None:
            depth_img=self.vis_result(place_pixel=place_pixel,pick_pixel=pick_pixel,img=depth_img.copy())
            encoded_vis_result=encode_image(depth_img)
            
        correct_message=[]
        
        
        text_correct_message="""
        I am providing you with the visualization result of your predicted pick-and-place action. In the image you can see a green circle which is your predicted picking point and a green arrow which points to your predicted move direction and a purple circle at the end of that arrow denoting the estimated placing point.\n
        """
        
        if self.depth_reasoning:
            # if the depth_reasoning is enabled, the depth image will be provided to the recal module.
            # Normally this will be false as the depth_reasoning method is not used in the paper due to much worse performance.
            text_correct_message+="""
            I am also providing you the visualization result of your predicted pick-and-place action on the corresponding depth image. In the depth image you can also see a green circle which is your predicted picking point and a green arrow which pointing to the your predicted move direction and a purple circle at the end of that arrow denoting the estimated placing point.\n 
            """
        
        
        # 1. Pick point approximity check    
        if check_directions_only:
            last_pick_point=None
        
        
        if last_pick_point is not None:
            # Check the distance between the predicted picking point and the last picking point and its symmetric point.
            pick_check=(abs(pick_pixel[0]-last_pick_point[0])>50) or (abs(pick_pixel[1]-last_pick_point[1])>50)
            pick_oppo_check=(abs(pick_pixel[0]-last_pick_point_oppo[0])>50) or (abs(pick_pixel[1]-last_pick_point_oppo[1])>50)
            
            
            if pick_check and pick_oppo_check:
                position_message="By calculation, the chosen picking point is not near the last picking point or its symmetric point, you can stick with this picking point."
                
            elif pick_check:
                position_message=f"By calculation, the chosen picking point is near the last picking point's symmetric point. The chosen picking point is [{pick_pixel[0]},{pick_pixel[1]}] and the last picking point's symmetric point is [{last_pick_point_oppo[0]},{last_pick_point_oppo[1]}] so the pick point is within 100 pixel range of that point, please choose another point to pick."
            else:
                position_message=f"By calculation, the chosen picking point is near the last picking point. The chosen picking point is [{pick_pixel[0]},{pick_pixel[1]}] and the last picking point is [{last_pick_point[0]},{last_pick_point[1]}] so the picking point is within 100 pixel range of that point, please choose another point to pick."
                

            text_correct_message+=position_message
        else:
            # If no last picking point is provided, skip the checking of the picking point (set the result to be true).
            pick_check=True
            pick_oppo_check=True
            
            
            
        # 2. Direction check    
        direction_pattern=r'Moving direction:.*?(\d+/\d+)'
        direction_match = re.search(direction_pattern, response_message)
        moving_direction = direction_match.group(1)
        numerator, denominator = moving_direction.split('/')
        moving_direction=float(numerator)/float(denominator)
        print(f"the moving_direction is {moving_direction} with type {type(moving_direction)}")
        
        # 2.a Calculate the direction from the center point to the picking point
        direction=self._cal_direction(center,pick_pixel)
        print(f"the cal_direction is {direction} with type {type(direction)}")
        
        
        # 2.b Check whether the predicted direction is close to the calculated direction (+/- 0.25*pi)
        difference=np.abs(moving_direction-direction)
        print(f"difference is {difference}")

        possible_directions=[]
        for i in range(1,9):
            possible_directions.append(i/4)
        
        possible_directions=np.array(possible_directions)
        possible_directions_diff=np.abs(possible_directions-direction)
        
        choice=np.argmin(possible_directions_diff)
        str_direction=self.directions[choice]
        
        left=choice-1
        right=choice+2
        
        if left<0:
            left=8+left
        if right>8:
            right=right-8
        if left<right:
            accept_direction_list=possible_directions[left:right]
            str_direction_list=self.directions[left:right]
        else:
            accept_direction_list=possible_directions[left:]+possible_directions[:right]
            str_direction_list=self.directions[left:]+self.directions[:right]
            
        str_direction_list=f"[{','.join(str_direction_list)}]"
        direction_check=(moving_direction in accept_direction_list) or difference<0.25
        
        print(direction_check)
        
        
        
        
        # 3. Get the result of both checks and generate the correction message w.r.t different check results.
        if direction_check and pick_check and pick_oppo_check:
            direction_message="\n By calculating the pick point you choose and the center point, the direction starting from the center point to the picking point is roughly "+str_direction+". The direction you predicted falls in the acceptable range."
        elif pick_check and pick_oppo_check:
            direction_message="\n The picking point is an acceptable choice as it's not near to the last picking point or its symmetric point. But by calculating the pick point you choose and center point, the direction starting from the center point to the picking point is roughly "+str_direction+". The direction you predicted doesn't fall in the acceptable range. Please use "+str_direction+"as the moving direction if you want to pick the same picking point."
        else:
            direction_message="\n The picking point is not an accept choice as it's near to last picking point or its symmetric point. The predicted moving direction is also incorrect."
            
        check_result=direction_check and pick_oppo_check and pick_check
        
        text_correct_message+=direction_message

        correction_message="""
        
        Based on the assistance of the previous calculation, do you think your predicted move will help flatten the fabric? If so, you can repeat your answer. If you don't think this move will help flatten the fabric, you should give a new prediction following the same output format.
        
        """
        
        text_correct_message+=correction_message
        
        
        
        text_content={
            "type":"text",
            "text":text_correct_message,
        }
        image_content={
            "type":"image_url",
            "image_url":{
                "url":f"data:image/jpeg;base64,{encoded_vis_result}",
                "detail":"high"
                }
        }
       
        correct_message.append(text_content)
        correct_message.append(image_content)
        return correct_message,check_result,direction_check
        
        
    def get_pick_place(self,messages,headers):
        """
        This function is used to get the picking point and placing point from GPT with correct format.
        Input:
            messages: the conversation history. Normally it will include the system prompt.
            headers: the headers for the GPT API
        Output:
            pick_point: the 3D coordinate of the picking point
            place_point: the 3D coordinate of the placing point
            pick_pixel: the pixel coordinate of the picking point
            place_pixel: the pixel coordinate of the placing point
            response_message: the response message from GPT
        """
        

        re_cal=True
        
        # 1. Deal with different types of errors from GPT
        while re_cal:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(**inputs, max_new_tokens=3000, do_sample=False)
                generation = generation[0][input_len:]

            
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            print(decoded)

            #response = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
                                
            pick_point,place_point,pick_pixel,_=self.response_process(decoded)
            
            if 'choices' in response.json():
                # GPT doesn't run into error.                
                response_message=response.json()['choices'][0]['message']['content']
                
                if pick_point is not None:
                    # GPT gives the correct output
                    re_cal=False
                    break
                else:
                    # GPT gives the output with format error 
                    # (The result doesn't contain pick point, direction, distance or not in the desired format)
                    re_cal=True
                    time.sleep(30)
                    format_error_message="The output given by you has format error, please output your result according to the given format."
                    
                    messages.append(
                        
                        {
                            "role":"assistant",
                            "content":response_message,
                        })
                        
                        
                    messages.append(    
                        {
                            "role":"user",
                            "content":[
                                
                                {"type":"text",
                                "text":format_error_message,
                                }
                            ]
                        }
                    )
                    
            else:
                # GPT runs into error. In our tests sometimes it's due to "inappropriate content" or "model error"
                re_cal=True
                time.sleep(30)
                format_error_message="I am passing you only two images with one being a fabric lying on the black surface and another is the depth image of that fabric with the cloth being in grayscale and the background being yellow (near brown). There's no inapproriate content. "
     
                messages.append(    
                    {
                        "role":"user",
                        "content":[
                            
                            {"type":"text",
                            "text":format_error_message,
                            }
                        ]
                    }
                )

        place_pixel=camera_utils.get_pixel_coord_from_world(place_point,(self.img_size,self.img_size),self.env)
        
        place_pixel=place_pixel.astype(int)
        
        return pick_point,place_point,pick_pixel,place_pixel,response_message
        
    def build_in_context_learning_prompt(self,
                                         demo_dir="./demo/demorun5",
                                         
                                         ):
    
        
        """
        This function is used to build the in-context learning prompt for the GPT.
        Note that we didn't use this in the paper as the in-context learning is not helping in our case (perhaps with some dataset creation techniques it will work).
        """
        # image_paths=[]
        input_image_paths=[]
        output_image_paths=[]
        
            
        
        
        examples="""\n\nHere are some examples for you to reference:\n\n"""
        
      
        
        examples+="\n\n-------This is the beginning of a full demonstration of 5 consecutive steps to flatten the fabric-------------\n\n"
        for i in range(5):
            # demo_path=osp.join(demo_dir,self.env_name)
            # # pc_path=demo_path+'_inittest'+str(i)+'.csv'
            # # pc=np.genfromtxt(pc_path, delimiter=',')
                
            # pc=self.trim_pc(pc,rate=5)
            # pc=np.round(pc,3)
            # pointcloud=self.obs_to_str(pc)

            
            input_text_path=osp.join(demo_dir,"demo_step_"+str(i)+'_corners.txt')
            input_image_path=osp.join(demo_dir,"processed_image_demo_step_"+str(i)+'.png')
            
            output_text_path=osp.join(demo_dir,"user_input_demo_step_"+str(i)+'.txt')
            output_image_path=osp.join(demo_dir,"Vis_result_demo_step_"+str(i)+'.png')
            
            
            
            with open (input_text_path,'r+') as p:
                input_message=p.read()
                
            with open (output_text_path,'r+') as p:
                output_message=p.read()
            
            
            example="\n##Step "+str(i+1)+"\n"
            
            
                
            str_input="\n##Input:\n"+input_message
            str_output="\n##Expected output:\n"+output_message+"\n (The visualization of the action is also provided)"
            
            example=example+str_input+"\n\n"+str_output
            
            examples=examples+example+"\n\n"
            
            
            input_image_paths.append(input_image_path)
            output_image_paths.append(output_image_path)
            
            
            # input_img={
            #     "type":"image_url",
            #     "image_url":{
            #         "url":f"data:image/jpeg;base64,{input_image_path}",
            #         "detail":"high"
            #     }
            # }
            
            # output_img={
            #     "type":"image_url",
            #     "image_url":{
            #         "url":f"data:image/jpeg;base64,{output_image_path}",
            #         "detail":"high"
            #     }
                
            # }
            
            # image_paths.append(input_img)
            # image_paths.append(output_img)
            
        examples+="\n\n-------This is the end of the 5 step demostration-------------\n\n"

        # uncomment here for debugging
        # with open("system_prompt_temp.txt","w+") as sys_promp: 
        #     sys_promp.write(system_prompt)
        return examples,input_image_paths,output_image_paths    
        # return examples, image_paths       
        
    def communicate(self,
                    headers,  
                    messages,
                    encoded_image,
                    corners,
                    center_point_pixel,
                    curr_coverage,
                    last_step_info,
                    goal_config=False,
                    direction_seg=8,
                    distance_seg=4):
        """
        This function is used to communicate with the GPT to get the actual pick-and-place action for implementation.
        Input:
            headers: the headers for the GPT API
            messages: the conversation history
            encoded_image: the encoded image to be sent to the GPT
            corners: the corners detected by the Shi-Tomasi corner detector
            center_point_pixel: the pixel coordinate of the center point of the fabric
            curr_coverage: the current coverage of the fabric
            last_step_info: the information of the last step (coverage, pick point, place point)
            direction_seg: the number of segments for the direction
            distance_seg: the number of segments for the distance
            
            
        Output:
            pick_point: the 3D coordinate of the picking point
            place_point: the 3D coordinate of the placing point
            messages: the updated conversation history
            last_step_info: the information of the this step
        """
        
        
        # 0. Setup the parameters
        content=[]
        corner_str_lst=[]
        
        for corner in corners:#perhaps do sth here
            corner_str=f"[{corner[0]},{corner[1]}]"
            corner_str_lst.append(corner_str)
            
        corners_str=f"{', '.join(corner_str_lst)}"
        print("test corners output: \n",corners_str)
        
        center_point_str=f"[{center_point_pixel[0]}, {center_point_pixel[1]}]"
        
        # 1. prepare the user prompt for input
        
        if last_step_info is None:
            # The first step
            coverage_message="This is the coverage of the cloth now:"+str(curr_coverage)+".\n"
            last_pick_point=None
            last_pick_point_oppo=None
            text_user_prompt={
            "type":"text",
            "text":coverage_message+"I am providing you the processed image of the current situation of the cloth to be smoothened. The blue points that you can see are the corners detected by Shi-Tomasi corner detector and here is their corresponding pixel:\n"+corners_str+"\n\nAnd the black point represents the center point of the cloth which is the center point of the cloth's bounding box. Its pixel is "+center_point_str+"\n\nJudging from the input image and the pixel coordinates of the corners and center point, please make the inference following the strategy and output the result using the required format."
        }
        else:
            # This step is not the first step
            
            coverage_change=curr_coverage-last_step_info['coverage']
            coverage_message="This is the coverage of the cloth now:"+str(curr_coverage)+".\n"
            
            last_pick_point=last_step_info['place_pixel']
            last_pick_point_oppo=[center_point_pixel[0]*2-last_pick_point[0],center_point_pixel[1]*2-last_pick_point[1]]
            last_pick_point_str=f'[{last_pick_point[0]},{last_pick_point[1]}]'
            last_pick_point_oppo_str=f'[{last_pick_point_oppo[0]},{last_pick_point_oppo[1]}]'
            
            
            text_user_prompt={
            "type":"text",
            "text":coverage_message+"I am providing you the processed image of the current situation of the cloth to be smoothened. The blue points that you can see are the corners detected by Shi-Tomasi corner detector and here is their corresponding pixel:\n"+corners_str+"\n\nAnd the black point represents the center point of the cloth which is the center point of the cloth's bounding box. Its pixel is "+center_point_str+"\n\n The red points are the pick point chosen last time and its symmetric point. Its pixel is "+last_pick_point_str+", and its symmetric point's pixel is "+last_pick_point_oppo_str+". It's advised to pick points that are not near those two points.\n\nJudging from the input image and the pixel coordinates of the corners and center point, please make the inference following the strategy and output the result using the required format."
        }
                        

        
                        
        
            
        # 1.b If the goal configuration is enabled, add the goal configuration information to the user prompt
        if goal_config:
            goal_config_information="\nTo help you with the task while planning, the image also has a white rectangular box around the cloth representing the goal configuration of the cloth which is the flattened cloth's outline. Please use it for reference"
            # goal_config_information could have the pixel values of the bounding box
            text_user_prompt["text"]+=goal_config_information
            
        content.append(text_user_prompt)
        print('message', content)
        image_user_prompt={
            "type":"image_url",
            "image_url":{
                "url":f"data:image/jpeg;base64,{encoded_image}",
                "detail":"high"
            }
        }
        #print('content', content)
        content.append(image_user_prompt)
        
        message={
            "role":"user",
            "content":content,
        }
        messages.append(message)
        
       
        # 2. Pass the user prompt and system prompt to GPT and get the response
        pick_point,place_point,pick_pixel,place_pixel,response_message=self.get_pick_place(messages=messages,headers=headers)
        
        messages.append(
            {
                "role":"assistant",
                "content":response_message,
            }
        )
        
        
        # 3. Use the recal (Evaluation) module to check the correctness of the predicted action
        
        if self.re_consider:
            steps=0 # set a counter to limit the number of recal steps
            recon_message,check_result,direction_check=self.recal(response_message=response_message,place_pixel=place_pixel,pick_pixel=pick_pixel,center=center_point_pixel,img=self._step_image, last_pick_point=last_pick_point,last_pick_point_oppo=last_pick_point_oppo)
            
            while not check_result:
                # If the response fails to pass the evalution, ask GPT to reconsider the action with added correction message.
                messages.append({
                    "role":"user",
                    "content":recon_message,
                })
                
                pick_point,place_point,pick_pixel,place_pixel,response_message=self.get_pick_place(messages=messages,headers=headers)
                messages.append(
                    {
                        "role":"assistant",
                        "content":response_message,
                    }
                )
                steps+=1
                if steps>=3 and direction_check:
                    # We check both direction and pick point approximity for at most 3 times. After 3 evalutions, we only check on the direction. 
                    break
                recon_message,check_result,direction_check=self.recal(response_message=response_message,place_pixel=place_pixel,pick_pixel=pick_pixel,center=center_point_pixel,img=self._step_image, last_pick_point=last_pick_point,last_pick_point_oppo=last_pick_point_oppo)
                
                
        
        
        # # 4. Visualize the result of the pick-and-place action (before actual interaction)
        # img=self.vis_result(place_pixel=place_pixel,pick_pixel=pick_pixel,img=self._step_image)
        # vis_result_path=osp.join(self.obs_dir,"Vis_result_"+self._specifier+".png")
        # cv.imwrite(vis_result_path,img)
        
        # 5. Update the last_step_info    
        last_step_info={
            "pick_pixel":pick_pixel,
            "place_pixel":place_pixel,
        }
        
        
        return pick_point,place_point,messages,last_step_info


    def communicate_with_depth(self,
                    headers,  
                    messages,
                    encoded_image,
                    encoded_depth_image,
                    corners,
                    center_point_pixel,
                    curr_coverage,
                    last_step_info,
                    direction_seg=8,
                    distance_seg=4):
        """
        Used for depth_reasoning method. Deprecated.
        """
        content=[]
        
        corner_str_lst=[]
        for corner in corners:
            corner_str=f"[{corner[0]},{corner[1]}]"
            corner_str_lst.append(corner_str)
            
        corners_str=f"{', '.join(corner_str_lst)}"
        # print("test corners output: \n",corners_str)
        
        center_point_str=f"[{center_point_pixel[0]}, {center_point_pixel[1]}]"
        
        
        
        if last_step_info is None:
            coverage_message="This is the coverage of the cloth now:"+str(curr_coverage)+".\n"
            text_user_prompt={
            "type":"text",
            "text":coverage_message+"I am providing you the processed image (image 1) of the current situation of the cloth to be smoothened. The blue points that you can see is the corners detected by Shi-Tomasi corner detector and here is their corresponding pixel:\n"+corners_str+"\n\nAnd the black point represents the center point of the cloth which is the center point of the cloth's bounding box. Its pixel is "+center_point_str+". \n\n I am also providing you the corresponding depth image (image 2) of the cloth."+"\n\nJudging from the input image , depth image and the pixel coordinates of the corners and center point, please making the inference following the strategy elaborated in the system prompt and output the result using the required format."
        }
        else:
            
            coverage_change=curr_coverage-last_step_info['coverage']
            
            coverage_message="This is the coverage of the cloth now:"+str(curr_coverage)+". With the action you predicted last time, the coverage of the fabric changed by "+str(coverage_change)+". If it's positive, the coverage increased otherwise the coverage drops.\n"
            
            last_pick_point=last_step_info['place_pixel']
            last_pick_point_str=f'[{last_pick_point[0]},{last_pick_point[1]}]'
            
            
            text_user_prompt={
            "type":"text",
            "text":coverage_message+"I am providing you the processed image (image 1) of the current situation of the cloth to be smoothened. The blue points that you can see is the corners detected by Shi-Tomasi corner detector and here is their corresponding pixel:\n"+corners_str+"\n\nAnd the black point represents the center point of the cloth which is the center point of the cloth's bounding box. Its pixel is "+center_point_str+". \n\n I am also providing you the corresponding depth image (image 2) of the cloth."+"\n\n The red point is the pick point chosen last time. Its pixel is "+last_pick_point_str+"\n\nJudging from the input image, depth image and the pixel coordinates of the corners and center point, please making the inference following the strategy elaborated in the system prompt and output the result using the required format."
        }
                        

        
                        
        
            
        
        if self.goal_config:
            goal_config_information="\nTo help you with the task while planning, the image also has a white rectangular box around the cloth representing the goal configuration of the cloth which is the flattened cloth's outline. Please use it for reference"
            # goal_config_information could have the pixel values of the bounding box
            text_user_prompt["text"]+=goal_config_information
            
        content.append(text_user_prompt)
        image_user_prompt={
            "type":"image_url",
            "image_url":{
                "url":f"data:image/jpeg;base64,{encoded_image}",
                "detail":"high"
            }
        }
        content.append(image_user_prompt)
        
        image_user_prompt_depth={
            "type":"image_url",
            "image_url":{
                "url":f"data:image/jpeg;base64,{encoded_depth_image}",
                "detail":"high"
            }
        }
        content.append(image_user_prompt_depth)
        
        message={
            "role":"user",
            "content":content,
        }
        messages.append(message)
        
        new_messages=[messages[0]]
        new_messages.append(message)
        
        
        
        pick_point,place_point,pick_pixel,place_pixel,response_message=self.get_pick_place(messages=messages,headers=headers)
        
        

        messages.append(
            {
                "role":"assistant",
                "content":response_message,
            }
        )

        
        
        
        if self.re_consider:
            recon_message=self.recal(place_pixel=place_pixel,pick_pixel=pick_pixel,center=center_point_pixel,img=self._step_image)
            messages.append({
                "role":"user",
                "content":recon_message,
            })
            pick_point,place_point,pick_pixel,place_pixel,response_message=self.get_pick_place(messages=messages,headers=headers)
            messages.append(
                {
                    "role":"assistant",
                    "content":response_message,
                }
            )
        
        
        
        self._step_image=self.vis_result(place_pixel=place_pixel,pick_pixel=pick_pixel,img=self._step_image)
        cv.imwrite(self.paths['processed vis image'],self._step_image)
        
        self.depth_image=self.vis_result(place_pixel=place_pixel,pick_pixel=pick_pixel,img=self.depth_image)
        cv.imwrite(self.paths['processed vis depth'],self._step_image)
        
        
        raw_vis_image=self.vis_result(place_pixel=place_pixel,pick_pixel=pick_pixel,img=cv.imread(self.paths["raw image"]))
        cv.imwrite(self.paths['raw vis image'],raw_vis_image)
        raw_vis_depth=self.vis_result(place_pixel=place_pixel,pick_pixel=pick_pixel,img=cv.imread(self.paths["raw depth"]))
        cv.imwrite(self.paths['raw vis depth'],raw_vis_depth)
        
        
            
        last_step_info={
            "pick_pixel":pick_pixel,
            "place_pixel":place_pixel,
        }
        
        
        return pick_point,place_point,messages,last_step_info


        

    def single_step(self, frames, last_step_info=None,corner_limit=10,need_box=True,  direction_seg=8, distance_seg=4, specifier="init"):
        """
        This is used to generate the pick-and-place action for a single step manually.
        We used this to generate the demonstration data for the ICL and finetuning.
        """
        
        default_pos=np.array([0.0,0.2,0.0]).squeeze()
        operation_height=0.1
        self._specifier=specifier
        
        
        self.paths={
            
            "raw image":osp.join(self.obs_dir,"raw_image_"+specifier+".png"),
            "raw depth":osp.join(self.obs_dir,"raw_depth_image_"+specifier+".png"),
            "processed image":osp.join(self.obs_dir,"processed_image_"+specifier+".png"),
            "processed depth":osp.join(self.obs_dir,"processed_depth_image_"+specifier+".png"),
            "raw vis image":osp.join(self.obs_dir,"Raw_vis_result_"+specifier+".png"),
            "raw vis depth":osp.join(self.obs_dir,"Raw_vis_result_depth_"+specifier+".png"),
            "processed vis image":osp.join(self.obs_dir,"Vis_result_"+specifier+".png"),
            "processed vis depth":osp.join(self.obs_dir,"Vis_result_depth_"+specifier+".png"),
            
        }
        
        
        
        # step 0.a : get and save obs before interaction:
        obs=self.env.get_rgbd()
        
        image=(obs[:,:,:-1]*255).astype(np.uint8)
        image_raw=Image.fromarray(image)
        image_raw.save(self.paths['raw image'])# Raw image (image 1)
        
        depth=np.round(obs[:,:,3:].squeeze(),3)
        self.depth=depth
        self.depth_image=self.map_depth_to_image(depth_array=depth)
        raw_depth_image=Image.fromarray(self.depth_image)
        raw_depth_image.save(self.paths['raw depth'])# Raw depth (image 2)
        
        self.pixel_coords=camera_utils.get_world_coords(rgb=image,depth=depth,env=self.env)[:,:,:-1]

        # step 0.b : to process the image and depth image

        image=self.aug_background(image,depth)
     
        img_path=self.save_obs(rgbd=obs,image=image,specifier=specifier)
        depth_image_path=self.save_obs(image=self.depth_image,specifier="depth_"+specifier)
        
        
        
        # step 0.c: get the corners on the image as well as the corner coordinates
        corners,img=self.get_corners_img(img_path=img_path,depth=depth,specifier=specifier,corner_limit=corner_limit)# The imgs will have corners marked at this stage
        
        # step 0.d: get the center point via bounding box of the fabric:
        center_point_pixel,preprocessed_img=self.get_center_point_bounding_box(img_path=img_path,depth=depth,need_box=need_box)
        
        # step 0.e: get last step's info here. 
        if (last_step_info is not None) and ('place_pixel' in last_step_info): 
            preprocessed_img=self.vis_result(img=preprocessed_img,place_pixel=last_step_info['place_pixel'])
            last_pick_point=last_step_info['place_pixel']
            last_pick_point_oppo=[center_point_pixel[0]*2-last_pick_point[0],center_point_pixel[1]*2-last_pick_point[1]]
            preprocessed_img=self.vis_result(img=preprocessed_img,place_pixel=last_pick_point_oppo)


        
        
        preprocessed_img_path=self.paths['processed image']
        self._step_image=preprocessed_img
        #cv.imwrite(preprocessed_img_path,preprocessed_img)# processed image (image_3)
        encoded_image=encode_image(preprocessed_img)
 
        
        

        
        info=self.env._get_info()

        curr_coverage=info["normalized_performance_2"]
        curr_coverage=np.round(curr_coverage,3)


        choice_path=osp.join(self.obs_dir,"user_input_"+specifier+".txt")           

        with open (choice_path,"w+") as p:
            p.write(f"coverage: {curr_coverage}")
            p.write("\n")
            p.write(f"center point: {[center_point_pixel[0],center_point_pixel[1]]}")

        print(f"------------Current Coverage is:{curr_coverage=}-----------------\n")
        
        print("These are the locations of the corners detected to help you grasp:\n")
        
        corner_str_lst=[]
        for corner in corners:#perhaps do sth here
            corner_str=f"[{corner[0]},{corner[1]}]"
            corner_str_lst.append(corner_str)
            
        corners_str=f"{', '.join(corner_str_lst)}"
        print(corners_str)
        

        

            
        user_input=input("List the place to pick using  \',\' as specifier:")
        print("\n")
        
        pick_point = user_input.split(",")
        
        pick_point = np.array(pick_point, dtype=int)
        pick_pixel=pick_point.copy()
        pick_point=camera_utils.find_nearest(self.pixel_coords,pick_point[1],pick_point[0])

        pick_point=self.pixel_coords[pick_point[0]][pick_point[1]]
        print(pick_point)
        
        
        
        self.directions=[]
        numerical_directions=[]
        self.distances=["0.1"]
        numerical_distances=[0.1]       
        for i in range(1,direction_seg+1):
            
            direction=str(i)+"/"+str(direction_seg//2)+"*pi"
            num_direction=float(i)/float(direction_seg//2)
            self.directions.append(direction)
            numerical_directions.append(num_direction)
        for j in range(1,distance_seg+1):
            distance=str(j/distance_seg)
            num_distance=float(j)/float(distance_seg)

            self.distances.append(distance)
            numerical_distances.append(num_distance)
            
        self.str_directions = f"[{', '.join(self.directions)}]"
        self.str_distances  = f"[{', '.join(self.distances)}]"
        
        print("These are the directions for you to choose:"+self.str_directions+"\n")
        chosen_direction=int(input("Please choose one from it and use the index (starting from 1) to indicate:"))-1
        print("\n")
        
        print("These are the distances for you to choose:"+self.str_distances+"\n")
        chosen_distance=int(input("Please choose one from it and use the index (starting from 1) to indicate:"))
        print("\n")
        
        choice={
            "Pick point:":pick_point,
            "Moving direction:":self.directions[chosen_direction],
            "Moving distance:":self.distances[chosen_distance],
        }
        
        
        with open(choice_path,"a+") as p:# print the choice and record them for later use
            for key,value in choice.items():
                item_str=f"{key}{value}\n"
                print(item_str)
                p.write(item_str)
            
        moving_direction=numerical_directions[chosen_direction]
        moving_distance=numerical_distances[chosen_distance]
        
        
        curr_config=self.env.get_current_config()
        dimx,dimy=curr_config['ClothSize']
        size=max(dimx,dimy)*self.env.cloth_particle_radius
        


        actual_direction=moving_direction*np.pi
        actual_distance=moving_distance*size

        delta_x=actual_distance*np.sin(actual_direction)
        delta_y=actual_distance*np.cos(actual_direction)


        place_point = pick_point.copy()
        place_point[0]+=delta_x
        place_point[2]+=delta_y
        
        place_pixel=camera_utils.get_pixel_coord_from_world(place_point,(self.img_size,self.img_size),self.env)
        place_pixel=place_pixel.astype(int)

        
        pre_pick_pos=pick_point.copy()
        pre_pick_pos[1]+=self.env.action_tool.picker_radius*2
        
        after_pick_pos=pick_point.copy()
        after_pick_pos[1]=operation_height
        
        pre_place_pos=place_point.copy()
        pre_place_pos[1]=operation_height
        
        
        action_sequence=[[pre_pick_pos, False],
                        [pick_point, True],
                        [after_pick_pos, True],
                        [pre_place_pos, True],
                        [pre_place_pos, False],
                        [default_pos, False]]
        
        
        for action in action_sequence:
            frames.extend(self.picker_step(target_pos=action[0],pick=action[1]))
            
        img=self.vis_result(place_pixel=place_pixel,pick_pixel=pick_pixel,img=self._step_image)
        vis_result_path=osp.join(self.obs_dir,"Vis_result_"+self._specifier+".png")
        cv.imwrite(vis_result_path,img)
        
        
        
            
        last_step_info={
            "pick_pixel":pick_pixel,
            "place_pixel":place_pixel,
            "coverage":curr_coverage,
        }
           
        
        info=self.env._get_info()        
        test_improvement=info["normalized_performance"]
        test_coverage=info["normalized_performance_2"]
        return frames,last_step_info,np.round(test_improvement,3),np.round(curr_coverage,3),np.round(test_coverage,3)
        
           

    
    def gpt_single_step(self,
                    info,
                    headers,
                    rgb, #RGB-D for current step
                    depth,
                    mask,
                    messages=[],
                    goal_config=False,
                    memory=True,
                    system_prompt_path=".system_prompts/COT_no_KP.txt",
                    need_box=True,
                    corner_limit=10,                     
                    last_step_info=None,
                    aug_background=True,
                    depth_reasoning=False,
                    direction_seg=8,
                    distance_seg=4,
                    specifier="init"):
        """
        This is used to generate the pick-and-place action for a single step using GPT.
        Note that the `communicate` function is called in this function to get the pick point and placing point,but no action is implemented in that function. The action is implemented in this function.
        Input:
            headers: the headers for the GPT API
            frames: the frames of the previous steps
            messages: the conversation history
            memory: whether to use memory (Whether to include previous steps' information in the conversation history)
            system_prompt_path: the path for the system prompt
            need_box: whether to draw the bounding box
            corner_limit: the limit of the number of corners to be detected
            last_step_info: the information of the last step
            aug_background: whether to augment the background
            depth_reasoning: whether to use depth reasoning (default is False and recommended not to use)
            direction_seg: the number of segments for the direction
            distance_seg: the number of segments for the distance
            specifier: the specifier for the step's related files
        Output:
            frames: the frames of the current step and previous steps
            last_step_info: the information of the current step
            test_improvement: the Normalized improvement of the current step comparing to starting configuration
            curr_coverage: the coverage of the current step before interaction
            test_coverage: the coverage of the current step after interaction
        """
        
        self._specifier=specifier
        self.depth_reasoning=depth_reasoning
        default_pos=np.array([0.0,0.2,0.0]).squeeze()
        operation_height=0.1
        
        # Do we need the follwing?
        # self.paths={
            
        #     "raw image":osp.join(self.obs_dir,"raw_image_"+specifier+".png"),
        #     "raw depth":osp.join(self.obs_dir,"raw_depth_image_"+specifier+".png"),
        #     "processed image":osp.join(self.obs_dir,"processed_image_"+specifier+".png"),
        #     "processed depth":osp.join(self.obs_dir,"processed_depth_image_"+specifier+".png"),
        #     "raw vis image":osp.join(self.obs_dir,"Raw_vis_result_"+specifier+".png"),
        #     "raw vis depth":osp.join(self.obs_dir,"Raw_vis_result_depth_"+specifier+".png"),
        #     "processed vis image":osp.join(self.obs_dir,"Vis_result_"+specifier+".png"),
        #     "processed vis depth":osp.join(self.obs_dir,"Vis_result_depth_"+specifier+".png"),
            
        # }


        # step 0.a : get and save obs before interaction:
        image = rgb # RGB in [0, 255]
        
        obs = np.concatenate([rgb, depth], axis=-1) #combine rgb and depth
        
        # Do we need the following?
        # self.pixel_coords=camera_utils.get_world_coords(rgb=image,depth=depth,env=self.env)[:,:,:-1]

        # step 0.b : to process the image and depth image
        
        if aug_background:
            save_mask(mask, 'gpt-mask', 'tmp')
            image=self.aug_background(image, mask)
        
        if depth_reasoning:
            self.depth_image=self.aug_background(self.depth_image,mask,color=[140,70,250])
            # self.depth_image=Image.fromarray(self.depth_image)
            
               
        #img_path=self.save_obs(rgbd=obs,image=image,specifier=specifier)
        #depth_image_path=self.save_obs(image=self.depth_image,specifier="depth_"+specifier)
        
        
        
        # step 0.c: get the corners on the image as well as the corner coordinates
        corners, img=self.get_corners_img(image, specifier=specifier,corner_limit=corner_limit)# The imgs will have corners marked at this stage
        print('corners', corners)
        # step 0.d: get the center point via bounding box of the fabric:
        save_colour(image, 'gpt-rgb-before', 'tmp')
        center_point_pixel,preprocessed_img=self.get_center_point_bounding_box(img,mask,need_box=need_box)
        save_colour(preprocessed_img, 'gpt-rgb', 'tmp')
        
        # step 0.e: get last step's info here. 
        if (last_step_info is not None) and ('place_pixel' in last_step_info): 
            preprocessed_img=self.vis_result(img=preprocessed_img,place_pixel=last_step_info['place_pixel'])
            last_pick_point=last_step_info['place_pixel']
            last_pick_point_oppo=[center_point_pixel[0]*2-last_pick_point[0],center_point_pixel[1]*2-last_pick_point[1]]
            preprocessed_img=self.vis_result(img=preprocessed_img,place_pixel=last_pick_point_oppo)


        
        
        #preprocessed_img_path=self.paths['processed image']
        self._step_image=preprocessed_img
        #cv.imwrite('./tmp/processed_img_for_gpt.png', preprocessed_img) # processed image (image_3)
        
        encoded_image=encode_image(preprocessed_img)
        
        
        if depth_reasoning:
            
            for corner in corners:
                cv.circle(self.depth_image,(corner[0],corner[1]),3,255,-1)# corners
            
            #cv.imwrite(depth_image_path,self.depth_image)
       
            _,self.depth_image=self.get_center_point_bounding_box(self.depth_image,depth=depth,need_box=need_box)# bounding box
            if (last_step_info is not None) and ('place_pixel' in last_step_info): 
                self.depth_image=self.vis_result(img=self.depth_image,place_pixel=last_step_info['place_pixel'])
            
            #cv.imwrite(self.paths['processed depth'],self.depth_image)
            encoded_depth_image=encode_image(self.depth_image) # processed depth (image_4)
        
        
        
        # step 1 : get coverage, improvement before action
        
        #info=self.env._get_info()
        improvement=info["evaluation"]["normalised_improvement"]
        improvement=np.round(improvement,3)
        curr_coverage=info["evaluation"]['normalised_coverage']
        curr_coverage=np.round(curr_coverage,3)
        
        
        # step 2 : Build system prompts


        self.directions=[]
        self.distances=[]       
        for i in range(1,direction_seg+1):
            
            direction=str(i)+"/"+str(direction_seg//2)+"*pi"
            self.directions.append(direction)
        for j in range(1,distance_seg+1):
            distance=str(j/distance_seg)
            self.distances.append(distance)
            
        self.str_directions = f"[{', '.join(self.directions)}]"
        self.str_distances  = f"[{', '.join(self.distances)}]"
        
        system_prompt_path = os.path.join('./controllers/gpt_fabric/', system_prompt_path)
        with open(system_prompt_path,"r") as file:
            system_prompt_text=file.read()
        
            
            
        system_prompt=[]         
        text_sys_prompt={
            "type":"text",
            "text":system_prompt_text
        }
        system_prompt.append(text_sys_prompt)
        
        if goal_config:
            system_prompt[0]["text"]+="Also, I have listed the image of the goal configuration of the fabric (cloth) for you to inference."
            image_sys_prompt={
                "type":"image_url",
                "image_url":{
                    "url":f"data:image/jpeg;base64,{self.goal_image}",
                    "detail":"high"
                }
            }
            system_prompt.append(image_sys_prompt)
                
        
        
        # step 2.1 Build in_context_learning examples if possible
        
        if self.in_context_learning: 
            # example_text,example_images=self.build_in_context_learning_prompt(demo_dir=self.demo_dir)
            # system_prompt[0]["text"]+=example_text
            example_text,input_images,output_images=self.build_in_context_learning_prompt(demo_dir=self.demo_dir)
            system_prompt[0]["text"]+=example_text
            
            for i in range(5):
                input_image={
                    "type":"image_url",
                    "image_url":{
                        "url":f"data:image/jpeg;base64,{encode_image(input_images[i])}",
                        "detail":"high"
                    }
                    
                }
                
                system_prompt.append(input_image)
                
                output_image={
                    "type":"image_url",
                    "image_url":{
                        "url":f"data:image/jpeg;base64,{encode_image(output_images[i])}",
                        "detail":"high"
                    }
                    
                }
                
                system_prompt.append(output_image)
                
                
                
            
            
            # last_coverage=None
    
        # step 2.2 : Prepared the final system prompt
        init_message={
            "role":"system",
            "content":system_prompt
        }
        
        
        
        if len(messages)==0:
            messages.append(init_message)
        
        
        # step 3 : Communicate with GPT to get the pick-and-place action
        
        if self.depth_reasoning:
            pick_point,place_point,messages,last_step_info=self.communicate_with_depth(headers=headers,
                                                messages=messages,
                                                curr_coverage=curr_coverage,
                                                last_step_info=last_step_info,
                                                encoded_image=encoded_image,
                                                encoded_depth_image=encoded_depth_image,
                                                corners=corners,
                                                center_point_pixel=center_point_pixel,
                                                direction_seg=direction_seg,
                                                distance_seg=distance_seg,
                                                )
            
        else: 
            pick_point,place_point,messages,last_step_info=self.communicate(headers=headers,
                                                goal_config=goal_config,
                                                messages=messages,
                                                curr_coverage=curr_coverage,
                                                last_step_info=last_step_info,
                                                encoded_image=encoded_image,
                                                corners=corners,
                                                center_point_pixel=center_point_pixel,
                                                direction_seg=direction_seg,
                                                distance_seg=distance_seg,
                                                )
        
        last_step_info["coverage"]=curr_coverage
        
        if not memory:
            # clear the previous steps' information if "memory" is not enabled
            messages=[]


        return messages,last_step_info, pick_point, place_point