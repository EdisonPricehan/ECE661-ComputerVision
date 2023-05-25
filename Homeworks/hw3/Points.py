# 2 squares in building image
building_points = [[200, 240], [370, 238], [375, 295], [215, 300]]
building_points1 = [[239, 390], [385, 391], [404, 653], [309, 651]]

# desired square points in physical plane that respect the aspect ratio of building image square
building_height_pixels = 180
building_width_pixels = 60
building_rectified_points = [[200, 240],
                             [200 + building_height_pixels, 240],
                             [200 + building_height_pixels, 240 + building_width_pixels],
                             [200, 240 + building_width_pixels]]

# 2 squares in projective removed building image
building_projective_removed_square1 = [[684, 1460], [936, 1549], [959, 1613], [704, 1519]]
building_projective_removed_square2 = [[354, 582], [607, 616], [636, 705], [385, 666]]

# 2 squares in nighthawks image
nighthawks_points = [[180, 75], [655, 78], [620, 805], [220, 803]]
nighthawks_points1 = [[101, 12], [733, 14], [677, 867], [162, 863]]

# desired square points in physical plane that respect the aspect ratio of nighthawks image plane square
nighthawks_height_pixels = 340
nighthawks_width_pixels = 600
nighthawks_rectified_points = [[180, 75],
                               [180 + nighthawks_height_pixels, 75],
                               [180 + nighthawks_height_pixels, 75 + nighthawks_width_pixels],
                               [180, 75 + nighthawks_width_pixels]]

# 2 squares in projective removed nighthawks image
nighthawks_projective_removed_square1 = [[154, 57], [688, 59], [778, 1006], [241, 1005]]
nighthawks_projective_removed_square2 = [[511, 1064], [716, 1065], [724, 1157], [520, 1156]]

card_points = [[254, 487], [1115, 609], [798, 1221], [178, 1242]]
card_points1 = [[285, 528], [1082, 632], [787, 1211], [203, 1228]]

card_height_pixels = 500
card_width_pixels = 500
card_rectified_points = [[254, 487],
                        [254 + card_height_pixels, 487],
                        [254 + card_height_pixels, 487 + card_width_pixels],
                        [254, 487 + card_width_pixels]]

card_projective_removed_square1 = [[322, 621], [1779, 978], [1797, 2745], [342, 2397]]
card_projective_removed_square2 = [[368, 688], [1734, 1008], [1750, 2688], [390, 2360]]


facade_points = [[286, 705], [405, 705], [375, 855], [251, 852]]
facade_points1 = [[387, 1033], [531, 1045], [486, 1352], [333, 1326]]

facade_height_pixels = 200
facade_width_pixels = 300
facade_rectified_points = [[286, 705],
                           [286 + facade_height_pixels, 705],
                           [286 + facade_height_pixels, 705 + facade_width_pixels],
                           [286, 705 + facade_width_pixels]]

facade_projective_removed_square1 = [[200, 495], [278, 484], [246, 561], [168, 570]]
facade_projective_removed_square2 = [[231, 231], [312, 217], [281, 283], [201, 297]]
