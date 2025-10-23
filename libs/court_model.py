court_width = 610
court_height = 1340
line_width = (0.04 / 13.4) * court_height 
net_y = court_height / 2
short_service_y = (1.98 / 13.4) * court_height
singles_offset_x = (0.46 / 6.1) * court_width
long_service_y_from_boundary = (0.76 / 13.4) * court_height

reference_points = {
    'p1': [0, net_y],               # net left
    'p2': [court_width, net_y],     # net right
    # short service line points
    'p3': [0 + line_width, net_y + short_service_y], # double short service line left
    'p4': [singles_offset_x, net_y + short_service_y], # single short service line left
    'p5': [court_width / 2 - line_width / 2, net_y + short_service_y + line_width], # center short service line left 
    'p55': [court_width / 2 + line_width / 2, net_y + short_service_y + line_width], # center short service line right
    'p6': [court_width - singles_offset_x, net_y + short_service_y], # single short service line right
    'p7': [court_width - line_width, net_y + short_service_y], # double short service line right
    # long service line points
    'p8': [0 + line_width, court_height - long_service_y_from_boundary], # double long service line left
    'p9': [singles_offset_x, court_height - long_service_y_from_boundary], # single long service line left
    'p10': [court_width / 2 - line_width / 2, court_height - long_service_y_from_boundary], # center long service line left
    'p100': [court_width / 2 + line_width /2, court_height - long_service_y_from_boundary], # center long service line left
    'p11': [court_width - singles_offset_x, court_height - long_service_y_from_boundary], # single long service line right
    'p12': [court_width - line_width, court_height - long_service_y_from_boundary], # double long service line right
    # back boundary points
    'p13': [0, court_height], # double back boundary left
    'p14': [singles_offset_x, court_height - line_width], # single back boundary left
    'p15': [court_width / 2 - line_width / 2, court_height - line_width], # center back boundary
    'p155': [court_width / 2 + line_width / 2, court_height - line_width], # center back boundary
    'p16': [court_width - singles_offset_x, court_height - line_width], # single back boundary right
    'p17': [court_width, court_height], # double back boundary right
}