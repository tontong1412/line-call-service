court_width = 6100
court_height = 13400
line_width = 40
net_y = court_height / 2
short_service_y = 1980
singles_offset_x = 460
long_service_y_from_boundary = 760

reference_points = {
    # net
    'n0': [0, net_y],              
    'n1': [court_width, net_y],     
    # short service line points
    's0': [line_width, net_y + short_service_y], # double short service line left
    's1': [singles_offset_x + line_width, net_y + short_service_y], # single short service line left
    's2': [court_width - singles_offset_x - line_width, net_y + short_service_y], # single short service line right
    's3': [court_width - line_width, net_y + short_service_y], # double short service line right
    # long service line points
    'l0': [line_width, court_height - long_service_y_from_boundary - line_width], # double long service line left
    'l1': [singles_offset_x + line_width, court_height - long_service_y_from_boundary - line_width], # single long service line left
    'l2': [court_width - singles_offset_x - line_width, court_height - long_service_y_from_boundary - line_width], # single long service line right
    'l3': [court_width - line_width, court_height - long_service_y_from_boundary - line_width], # double long service line right
    # back boundary points
    'b0': [line_width, court_height - line_width], # double back boundary left
    'b1': [singles_offset_x + line_width, court_height - line_width], # single back boundary left
    'b2': [court_width - singles_offset_x - line_width, court_height - line_width], # single back boundary right
    'b3': [court_width -line_width, court_height - line_width], # double back boundary right
}

reference_lines = {
    # vertical lines d=double, s=single, c=center
    'd0': [[0, 0], [0, court_height]],
    'd1': [[court_width, 0], [court_width, court_height]],
    's0': [[singles_offset_x, 0], [singles_offset_x, court_height]],
    's1': [[court_width - singles_offset_x, 0], [court_width - singles_offset_x, court_height]],
    'c0': [[(court_width - line_width) / 2, 0], [(court_width - line_width) / 2, court_height]],
    'c1': [[(court_width + line_width) / 2, 0], [(court_width + line_width) / 2, court_height]],
    # horizontal lines, b=back, l=long, sh=short, n=net
    'b0': [[0, 0], [court_width, 0]],
    'b1': [[0, court_height], [court_width, court_height]],
    'l0': [[0, long_service_y_from_boundary], [court_width, long_service_y_from_boundary]],
    'l1': [[0, court_height - long_service_y_from_boundary], [court_width, court_height - long_service_y_from_boundary]],
    'sh0': [[0, net_y - short_service_y], [court_width, net_y - short_service_y]],
    'sh1': [[0, net_y + short_service_y], [court_width, net_y + short_service_y]],
    'n': [[0, net_y], [court_width, net_y]]
}