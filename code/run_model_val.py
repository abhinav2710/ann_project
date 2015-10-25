import numpy as np
import caffe
import sys
import os
import argparse

def make_arguments():
    parser = argparse.ArgumentParser(description="Run on images")

    # adding arguments
    parser.add_argument('deploy_prototxt')
    parser.add_argument('image_list')
    parser.add_argument('image_basepath')
    parser.add_argument('model_file')
    parser.add_argument('mean_file')
    parser.add_argument('gpu_id', type=int)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = make_arguments()

    caffe_root = '/home/abhinav/Softwares/caffe/'
    caffe.set_device(args.gpu_id)
    caffe.set_mode_gpu()
    images_labels = np.loadtxt(args.image_list, str)
    image_list = [i[0] for i in images_labels]
    image_labels = [i[1] for i in images_labels]
    net = caffe.Net(args.deploy_prototxt,
                    args.model_file,
                    caffe.TEST)

    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( args.mean_file , 'rb' ).read()
    blob.ParseFromString(data)
    mean_array = np.array( caffe.io.blobproto_to_array(blob) )
    mean_array = mean_array[0]

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', mean_array.mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    net.blobs['data'].reshape(10,3,227,227)
    
    predictions = []
    for image_name in image_list:
        print(image_name)
        full_path = os.path.join(os.path.dirname(args.image_basepath), image_name)

        image = caffe.io.load_image(full_path)
        proc_image = transformer.preprocess('data', image)
        proc_image = np.swapaxes(proc_image, 0, 2)
        proc_image = proc_image.reshape(1, 227, 227, 3)
        proc_image = caffe.io.oversample(proc_image, (227, 227))
        proc_image = np.swapaxes(proc_image, 1, 3)
        net.blobs['data'].data[...] = proc_image
        out = net.forward()
        scores = out['prob']
        scores = np.mean(scores, axis=0)
        ranks = scores.argsort()[-5:]
        print(ranks)
        predictions.append(ranks)
    predictions = np.array(predictions)
    np.savetxt('OUT.csv',predictions)
    
        
