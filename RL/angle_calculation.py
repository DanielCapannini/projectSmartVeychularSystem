def angle_calculation(image):
    imageP = preprocess_image(image)
    edges = cv2.Canny(imageP, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    min_angle = 75
    max_angle = 105
    min_angle_rad = np.deg2rad(min_angle)
    max_angle_rad = np.deg2rad(max_angle)
    theta_sum = 0
    line_count = 0
    for line in lines:
        _, theta = line[0]
        if min_angle_rad <= theta <= max_angle_rad:
            theta_sum += theta
            line_count += 1
    if  line_count > 0:
        theta_mean = theta_sum / line_count
        theta_mean_deg = np.rad2deg(theta_mean) 
        if theta_mean_deg > 90:
            return theta_mean_deg+5
        if theta_mean_deg < 90:
            return theta_mean_deg-2
        return theta_mean_deg
    return None