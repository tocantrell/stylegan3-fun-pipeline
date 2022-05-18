import sys
import os
import cv2
import numpy as np
import tempfile
import imageio
from character_projector import run_projection, generate_style_mix, generate_sprite_w_space

def clean_sprite(img_loc,
                 out_name,
                 gif_gen=True):

    try:
        img = cv2.imread(img_loc)
    except:
        print("Problem reading in generated sprite image.")
        sys.exit()

    dest = img_loc.split('/')[:-2]
    dest = ''.join(dest)+'/final'
    os.makedirs(dest, exist_ok=True)

    x = [0,32,64]
    y = [0,32,64,96]

    output_img = np.zeros((128,128,4), np.uint8)
    #output_img[:] = (255, 255, 255, 0)

    for i in y:
        for j in x:

            white_img = np.zeros((96,96,3), np.uint8)
            white_img[:] = (255, 255, 255)

            white_img[32:64,32:64] = img[i:i+32,j:j+32]

            hh, ww = white_img.shape[:2]

            # threshold on black
            # Define lower and uppper limits of what we call "white-ish"
            lower = np.array([200, 200, 200])
            upper = np.array([255, 255, 255])

            # Create mask to only select black
            white_thresh = cv2.bilateralFilter(white_img, 11, 200, 20)
            thresh = cv2.inRange(white_thresh, lower, upper)

            # invert mask so shapes are white on black background
            thresh_inv = 255 - thresh

            # get the largest contour
            contours = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            #big_contour = max(contours, key=cv2.contourArea)

            # draw white contour on black background as mask
            mask = np.zeros((hh,ww), dtype=np.uint8)
            cv2.drawContours(mask, contours, -1, (255,255,255), cv2.FILLED)

            # invert mask so shapes are white on black background
            mask_inv = 255 - mask

            # create new (blue) background
            bckgnd = np.full_like(white_img, (255,0,0))

            # apply mask to image
            image_masked = cv2.bitwise_and(white_img, white_img, mask=mask)

            # apply inverse mask to background
            bckgnd_masked = cv2.bitwise_and(bckgnd, bckgnd, mask=mask_inv)

            # add together
            result = cv2.add(image_masked, bckgnd_masked)
            #Add a black line around each sprite for style
            #cv2.drawContours(result, contours, -1, (0,0,0), thickness=1)

            image_bgr = result.copy()
            # get the image dimensions (height, width and channels)
            h, w, c = image_bgr.shape
            # append Alpha channel -- required for BGRA (Blue, Green, Red, Alpha)
            image_bgra = np.concatenate([image_bgr, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
            # create a mask where white pixels ([255, 255, 255]) are True
            white = np.all(image_bgr == [255, 0, 0], axis=-1)
            # change the values of Alpha to 0 for all the white pixels
            image_bgra[white, -1] = 0

            #Put the cleaned transparent sprite onto the final img
            output_img[i:i+32,j:j+32] = image_bgra[32:64,32:64]

    sprite_dest = dest+"/"+out_name+".png"
    cv2.imwrite(sprite_dest, output_img)

    if gif_gen == True:
        os.makedirs(dest+"/gif", exist_ok=True)
        generate_gif(sprite_dest,dest=dest+"/gif",out_name=out_name+".gif")

    return sprite_dest

def generate_gif(input_file,
                 dest='gif',
                 out_name='output.gif',
                 sprite_type='character'):

    if sprite_type == 'character': #May add more animations in future
        #animate_list = [2,3,4,7,8,9,17,18,19,12,13,14]
        animate_list = [3,2,3,4,3,7,8,9,8,17,18,19,18,12,13,14,13]

    #for index in index_list:
    with tempfile.TemporaryDirectory() as temp_folder:

        os.makedirs(dest, exist_ok=True)

        im =  cv2.imread(input_file, cv2.IMREAD_UNCHANGED)

        imgheight=im.shape[0]
        imgwidth=im.shape[1]


        if (imgheight%32 == 0) & (imgwidth%32 == 0):

            y1 = 0
            M = 32
            N = 32

        else:
            print("File not correct size to convert to GIF:")
            print(file)
            sys.exit()

        k = 0

        for y in range(0,imgheight,M):
            k+=1
            for x in range(0, imgwidth, N):
                k+=1
                y1 = y + M
                x1 = x + N
                tiles = im[y:y+M,x:x+N]
                break_list = [2,3,4,5]
                #if (len(np.unique(tiles)) in break_list):
                #    continue
                #else:
                cv2.rectangle(im, (x, y), (x1, y1), (0, 255, 0))

                if tiles.shape[2]==4:
                    B, G, R, A = cv2.split(tiles)
                    alpha = A / 255

                    R = (255 * (1 - alpha) + R * alpha).astype(np.uint8)
                    G = (255 * (1 - alpha) + G * alpha).astype(np.uint8)
                    B = (255 * (1 - alpha) + B * alpha).astype(np.uint8)

                    tiles = cv2.merge((B, G, R))

                else:
                    tiles[np.all(tiles == (120,180,150), axis=-1)] = (255,255,255)
                    tiles[np.all(tiles == (128,195,120), axis=-1)] = (255,255,255)
                    tiles[np.all(tiles == (0,255,0), axis=-1)] = (255,255,255)

                #cv2.imwrite(temp_folder+'/'+str(k) + ".png",tiles)
                cv2.imwrite(temp_folder+"/"+str(k) + ".png",tiles)

        frames = []
        for i in animate_list:
            mid_frame = imageio.imread(temp_folder+"/"+str(i) + ".png")
            frames.append(mid_frame)
        imageio.mimsave(dest+"/"+out_name, frames, format='GIF', duration=0.3)

def generate_face(sprite_loc,
                out_name,
                network_pkl_face,
                w_base_face,
                truncation_psi=0.6,
                dest='face',
                pkl='models/anime_faces_latest.pkl',
                num_steps=300):

    os.makedirs(dest, exist_ok=True)
    os.makedirs(dest+"/w_space", exist_ok=True)
    os.makedirs(dest+"/pre_style_mix", exist_ok=True)
    os.makedirs(dest+"/final", exist_ok=True)

    #Load sprite and convert transparancy to white
    sprite = cv2.imread(sprite_loc, cv2.IMREAD_UNCHANGED)
    trans_mask = sprite[:,:,3] == 0
    sprite[trans_mask] = [255, 255, 255, 255]
    sprite = cv2.cvtColor(sprite, cv2.COLOR_BGRA2BGR)

    #sprite = sprite[0:22,37:59]
    sprite = sprite[0:20,36:60]
    sprite = cv2.resize(sprite,(256,256),interpolation=cv2.INTER_AREA)

    seed = np.random.randint(3000000)

    out_loc_sprite = dest+"/pre_style_mix/input_"+out_name+".png"
    out_loc_w = dest+"/w_space/"+out_name+".npy"
    cv2.imwrite(out_loc_sprite,sprite)

    out_loc_premix = dest+"/pre_style_mix/"+out_name+".png"
    out_loc_final = dest+"/final"

    w_loc = run_projection(network_pkl=network_pkl_face,
                           target_fname=out_loc_sprite,
                           outloc=out_loc_premix,
                           w_outloc=out_loc_w,
                           seed=seed,
                           num_steps=num_steps)
    col_expression = [6]
    col_color = [10,11,12,13,14,15]
    col_style = [0,8,9,10,11,12,13,14,15]
    col_pose = [1]#[1]
    col_ws_styles = [col_expression,col_expression,col_pose,col_color]
    style_names = ['sad','happy','sideways','blue']
    style_loc_list = ['models/face_w_sad2.npy',
                      'models/face_w_wink.npy',
                      'models/face_w_sideways2.npy',
                      'models/face_w_blue.npy']

    generate_style_mix(network_pkl=network_pkl_face,
                       outloc=out_loc_final,
                       name=out_name,
                       w_input_loc=w_loc,
                       w_base_loc=w_base_face,
                       col_ws=[0,2,6,7,8],#1=pose 2=height/location 3=foreground obstruction 4&5=hair/face shape 6=expression 7=eye shape/expression and bangs/hair over cheeks 8=pupil size 9=hightlighs/shadows 10=line art/hue 11=eyebrows 12=saturation 13=eye color 14=skin color/saturation 15=hair color
                       truncation_psi=truncation_psi,
                       style_run=True,
                       style_names=style_names,
                       style_w_input_locs=style_loc_list,
                       style_col_ws=col_ws_styles)

def generate_sprite(
        network_pkl,
        out_name,
        w_base_sprite,
        dest='sprite'):

    """
    Function for generating a random sprite image
    and then mixing it with a pre-made style
    """

    os.makedirs(dest, exist_ok=True)
    os.makedirs(dest+"/w_space", exist_ok=True)
    os.makedirs(dest+"/pre_style_mix", exist_ok=True)
    os.makedirs(dest+"/post_style_mix", exist_ok=True)

    generate_sprite_w_space(network_pkl,
                            out_name,
                            dest_w=dest+"/w_space",
                            dest_sprite=dest+"/pre_style_mix")

    out_loc = dest+"/post_style_mix"
    w_loc = dest+"/w_space/"+out_name+".npy"

    img_loc = generate_style_mix(network_pkl,
                               outloc=out_loc,
                               name=out_name,
                               w_input_loc=w_loc,
                               w_base_loc=w_base_sprite,
                               col_ws=[5,6],#[3,5,6]
                               truncation_psi=1.)

    return img_loc

if __name__ == "__main__":

    out_name = sys.argv[1]
    truncation_psi = sys.argv[2]

    network_pkl_sprite = 'models/together_sprite_sg2.pkl'
    network_pkl_face ='models/anime_faces_latest.pkl'
    if out_name == 'hero':
        w_base_sprite = 'models/sprite_girl_w.npy'
        w_base_face = 'models/wide_eye_w.npy' #face_magical_w.npy
    elif out_name == 'villain':
        w_base_sprite = 'models/sprite_woman_w.npy'
        w_base_face = 'models/narrow_eye_w.npy'
    print("Generating sprites..")
    img_loc = generate_sprite(network_pkl_sprite, out_name, w_base_sprite)
    sprite_dest = clean_sprite(img_loc, out_name)
    print("Character sprites generated!")
    print("Generating faces..")
    generate_face(sprite_dest, out_name, network_pkl_face, w_base_face,truncation_psi)
    print("Faces generated!")
    print("Process finished.")
