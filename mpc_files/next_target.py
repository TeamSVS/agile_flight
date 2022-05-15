def next_target(depth):
    X, Y = 0, 0
    front_camera = depth
    
    self.arbitrator_outputs_X.append(X) #Needed to find the previous outputs of the arbitrator so to be able to output a corrected position given the previous outputs 
    self.arbitrator_outputs_Y.append(-Y)

    # If the previous coordinates were not (0, 0), then reflect them to get back to the center
    new_X = -np.mean(self.arbitrator_outputs_X)
    new_Y = -np.mean(self.arbitrator_outputs_Y)

    if front_camera is not None:
        camera_height = front_camera.shape[0]
        camera_width = front_camera.shape[1]
        if new_X < -(camera_width / 2):
            new_X = -(camera_width / 2)
        elif new_X > camera_width / 2:
            new_X = camera_width / 2
        if new_Y < -(camera_height / 2):
            new_Y = -(camera_height / 2)
        elif new_Y > camera_height / 2:
            new_Y = camera_height / 2


    return self.cart2pol(new_X/2, new_Y/2)
