from tracknetv3.utils.general import generate_frames
import cv2
from datetime import datetime
from tracknetv3.predict import track_ball_position

test_court_coord = {'net_line': [[337.0, 471.0], [945.0, 472.0]], 'short_service_line_top': [[356.7099914550781, 426.9200134277344], [925.5, 427.8999938964844]], 'short_service_line_bottom': [[314.3699951171875, 521.6099853515625], [967.3900146484375, 522.6199951171875]], 'long_service_line_top': [[389.3699951171875, 353.8800048828125], [893.1699829101562, 354.80999755859375]], 'long_service_line_bottom': [[257.29998779296875, 649.25], [1023.8400268554688, 650.260009765625]], 'center_line_top': [[641.4099731445312, 342.19000244140625], [641.25, 427.4100036621094]], 'center_line_bottom': [[640.7899780273438, 679.5], [641.0800170898438, 522.1199951171875]], 'singles_left_line': [[432.010009765625, 341.79998779296875], [303.8800048828125, 679.0800170898438]], 'singles_right_line': [[850.6400146484375, 342.5899963378906], [977.280029296875, 679.9199829101562]], 'outer_boundary_top': [[394.79998779296875, 341.7300109863281], [887.7899780273438, 342.6600036621094]], 'outer_boundary_bottom': [[244.0, 679.0], [1037.0, 680.0]], 'outer_boundary_left': [[394.79998779296875, 341.7300109863281], [244.0, 679.0]], 'outer_boundary_right': [[887.7899780273438, 342.6600036621094], [1037.0, 680.0]]}
video_file = 'test_video.mp4'

# test_court_coord = {'net_line': [[334.0, 474.0], [948.0, 475.0]], 'short_service_line_top': [[353.3800048828125, 429.9100036621094], [928.3699951171875, 431.0400085449219]], 'short_service_line_bottom': [[311.7900085449219, 524.530029296875], [970.47998046875, 525.3400268554688]], 'long_service_line_top': [[385.5799865722656, 356.6700134277344], [895.72998046875, 357.9700012207031]], 'long_service_line_bottom': [[255.97000122070312, 651.5], [1026.9000244140625, 651.6799926757812]], 'center_line_top': [[641.1099853515625, 345.1199951171875], [641.530029296875, 430.4800109863281]], 'center_line_bottom': [[642.75, 681.0], [641.989990234375, 524.9400024414062]], 'singles_left_line': [[428.739990234375, 344.55999755859375], [303.45001220703125, 681.0]], 'singles_right_line': [[852.77001953125, 345.67999267578125], [980.25, 681.0]], 'outer_boundary_top': [[390.95001220703125, 344.4599914550781], [890.2899780273438, 345.7799987792969]], 'outer_boundary_bottom': [[243.0, 681.0], [1040.0, 681.0]], 'outer_boundary_left': [[390.95001220703125, 344.4599914550781], [243.0, 681.0]], 'outer_boundary_right': [[890.2899780273438, 345.7799987792969], [1040.0, 681.0]]}
# video_file = 'test_video1.mp4'


# video_file = 'test_parainen_720.mp4'
# test_court_coord={'net_line': [[464.0, 490.0], [824.0, 493.0]], 'short_service_line_top': [[492.7200012207031, 475.9700012207031], [793.469970703125, 478.55999755859375]], 'short_service_line_bottom': [[421.1199951171875, 510.95001220703125], [869.4400024414062, 514.489990234375]], 'long_service_line_top': [[528.77001953125, 458.3500061035156], [755.0399780273438, 460.3900146484375]], 'long_service_line_bottom': [[210.1999969482422, 614.02001953125], [1090.47998046875, 619.0]], 'center_line_top': [[641.97998046875, 456.9100036621094], [643.5800170898438, 477.2699890136719]], 'center_line_bottom': [[658.25, 664.530029296875], [646.3599853515625, 512.72998046875]], 'singles_left_line': [[550.0900268554688, 456.07000732421875], [195.1999969482422, 662.3900146484375]], 'singles_right_line': [[733.52001953125, 457.739990234375], [1112.280029296875, 666.6300048828125]], 'outer_boundary_top': [[533.72998046875, 455.92999267578125], [749.739990234375, 457.8900146484375]], 'outer_boundary_bottom': [[112.0, 662.0], [1192.0, 667.0]], 'outer_boundary_left': [[533.72998046875, 455.92999267578125], [112.0, 662.0]], 'outer_boundary_right': [[749.739990234375, 457.8900146484375], [1192.0, 667.0]]}

cap = cv2.VideoCapture(video_file)




width, height = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# w_scaler, h_scaler = width / WIDTH, height / HEIGHT
# img_scaler = (w_scaler, h_scaler)

frame_list = generate_frames(video_file)
filename = datetime.now().strftime("%m_%d_%Y_%H:%M:%S")

# for i in range(625, len(frame_list), 1):
#     cv2.imshow(f"output_frames/frame_{i}.jpg", frame_list[i])
#     cv2.waitKey(0)

# cv2.imshow("Badminton Court Frame", frame_list[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

track_ball_position(
    frame_list,
    width,
    height,
    filename,
    batch_size=8,
    court_coord=test_court_coord,
    court_corners='court_corners',
)