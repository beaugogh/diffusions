import numpy as np
import os
import cv2
import random

def SH_basis(normal):
    '''
        get SH basis based on normal
        normal is a Nx3 matrix
        return a Nx9 matrix
        The order of SH here is:
        1, Y, Z, X, YX, YZ, 3Z^2-1, XZ, X^2-y^2
    '''
    '''
                ^ z  > y
                |  /
                | /
                |/            
        --------------------------> x
                |
                |
    '''
    numElem = normal.shape[0]

    norm_X = normal[:,0]
    norm_Y = normal[:,1]
    norm_Z = normal[:,2]

    sh_basis = np.zeros((numElem, 9))
    att= np.pi*np.array([1, 2.0/3.0, 1/4.0])
    sh_basis[:,0] = 0.5/np.sqrt(np.pi)*att[0]

    sh_basis[:,1] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_Y*att[1]
    sh_basis[:,2] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_Z*att[1]
    sh_basis[:,3] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_X*att[1]

    sh_basis[:,4] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_Y*norm_X*att[2]
    sh_basis[:,5] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_Y*norm_Z*att[2]
    sh_basis[:,6] = np.sqrt(5)/4/np.sqrt(np.pi)*(3*norm_Z**2-1)*att[2]
    sh_basis[:,7] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_X*norm_Z*att[2]
    sh_basis[:,8] = np.sqrt(15)/4/np.sqrt(np.pi)*(norm_X**2-norm_Y**2)*att[2]
    return sh_basis


def generate_2d_normals(begin_angle, end_angle, clockwise=True, num=10):
    '''
                 (b)
                  +
                  |
                  |
        ---------- ----------+ (a) (ref angle)
                  |
                  |
    '''
    arr = np.linspace(begin_angle, end_angle, num=num)
    sign = -1 if clockwise else 1
    pairs = []
    for i in arr:
        a = np.cos(i)
        b = np.sin(i) * sign
        pairs.append((a, b))
        
    return pairs
    

def generate_3d_normals(begin_angle, end_angle, clockwise=True, num=10, null_dim='y'):
    pairs = generate_2d_normals(begin_angle, end_angle, clockwise, num)
    result = []
    for a,b in pairs:
        if null_dim == 'y':
            result.append((a, 0, b))
        elif null_dim == 'x':
            result.append((0, a, b))
        elif null_dim == 'z':
            result.append((a, b, 0))
        else:
            raise Error('null_dim is ill defined')
    
    return np.reshape(result, (-1, 3))


def write_sh_to_files():
    # lightings_1
    # num_points = 240
    # normals_xy = generate_3d_normals(0.5*np.pi, -1.5*np.pi, clockwise=True, num=num_points, null_dim='z')
    # normals_yz = generate_3d_normals(-1*np.pi, 0.5*np.pi, clockwise=True, num=num_points, null_dim='x')
    # normals_xz = generate_3d_normals(0.5*np.pi, 2*np.pi, clockwise=True, num=num_points, null_dim='y')
    # normals = np.concatenate((normals_xy, normals_yz, normals_xz))

    # lightings_2
    # num_points = 120
    # normals = generate_3d_normals(np.pi, -1*np.pi, clockwise=True, num=num_points, null_dim='z')

    # lightings_3
    # num_points = 120
    # normals = generate_3d_normals(-1*np.pi, np.pi, clockwise=True, num=num_points, null_dim='x')

    # lightings_4
    num_points = 120
    normals = generate_3d_normals(-0.5*np.pi, 1.5*np.pi, clockwise=True, num=num_points, null_dim='y')

    shs = SH_basis(normals)
    for i, sh in enumerate(shs):
        print(i)
        np.savetxt('rotate_light_{:02d}.txt'.format(i), sh, delimiter='\n')



def images_to_video(fps=8, src='data/', output_filename='output_vid.mp4'):
    out_dir = os.path.join(output_filename)
    imgs = sorted(os.listdir(src))
    first_img = cv2.imread(os.path.join(src, imgs[0]))
    h, w, c = first_img.shape
    size = (w, h)
    print('video img size: ', size)
    out = cv2.VideoWriter(out_dir,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for filename in imgs:
        print(filename)
        img = cv2.imread(os.path.join(src, filename))
        out.write(img)
   
    out.release()


def rename_images(src='data/'):
    imgs = sorted(os.listdir(src))
    for filename in imgs:
        name = filename.split('.')[0]
        parts = name.split('_')
        head = parts[0]
        num = int(parts[-1])
        name_str = '{}_{:03d}.jpg'.format(head, num)
        os.rename(os.path.join(src, filename), os.path.join(src, name_str))
        print(name, name_str)


def resize_images(src='data/', width=800, height=1000):
    imgs = sorted(os.listdir(src))
    for i, filename in enumerate(imgs):
        print(i, filename)
        img_dir = os.path.join(src, filename)
        img = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED)
        dim = (width, height) 
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
        cv2.imwrite(img_dir, resized)


def custom_images_to_video(src='data/', output_filename='output_vid.mp4', span=180, total=720, fps=72):
    out_dir = os.path.join(output_filename)
    imgs = sorted(os.listdir(src))
    first_img = cv2.imread(os.path.join(src, imgs[0]))
    h, w, c = first_img.shape
    size = (w, h)
    print('video img size: ', size)
    out = cv2.VideoWriter(out_dir,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    
    portraits = {}
    for i, filename in enumerate(imgs):
        parts = filename.split('.')[0].split('_')
        name = parts[0]
        if name in portraits:
            portraits[name].append(filename)
        else:
            portraits[name] = [filename]

    keys = list(portraits.keys())
    random.shuffle(keys)
    print(keys)
    keys = ['zuck', 'kobe', 'bono', 'webber', 'clinton', 'pitt', 'rudd', 'dafoe', 'justin', 'curry', 
'gates', 'lebron', 'hopkins', 'watts', 'eastwood', 'merkel', 'jolie', 'deniro', 'musk', 
'zidane', 'bale', 'sting', 'natalie', 'jacobs',
'emma', 'daniel','rupert',  'washington', 'waltz', 'obama', 'amy', 'helen', 'mcconaughey', 'ledger', 
'streep', 'rihanna', 'anne', 'toni', 'jayz', 'cooper', 'blanchett', 'hillary', 'clooney', 'kloss', 
'adele', 'woody', 'murray', 'taylor', 'jane']

    start_indx = 0
    end_indx = start_indx + span
    prev_name = ''
    for k in keys:
        for filename in portraits[k]:
            msg = 'ignored'
            parts = filename.split('.')[0].split('_')
            name = parts[0]
            num = int(parts[1])

            if num >= start_indx and num < end_indx:
                img = cv2.imread(os.path.join(src, filename))
                out.write(img)
                msg = 'used'

            if num == end_indx - 1:
                msg = 'reset'
                prev_name = name
                if num == total-1:
                    start_indx = 0
                    end_indx = start_indx + span
                else:
                    start_indx = end_indx
                    end_indx = start_indx + span
                    end_indx = total if end_indx > total-1 else end_indx

                break

            
            print(filename, start_indx, end_indx, msg)
        
   
    out.release()



# write_sh_to_files()
# rename_images()
# resize_images(src='hopkins/')
# custom_images_to_video()
images_to_video(fps=30, src='jayz/', output_filename='output_vid.mp4')
