import numpy as np


def TrackMatrix(observation, printing=False):
    # returns truth table of the location of the track ahead of the car
    y = 64
    x = 96

    isTrack = np.zeros((y, x))
    to_print = ""
    # Loop over all pixels ahead of the car
    for i in range(y):
        for j in range(x):
            # if not green (if track pixel)
            if observation[i][j][1] <= 200:
                # if not black
                if sum(observation[i][j]) >= 30:
                    isTrack[i][j] = 1
            # Visualization
            if isTrack[i][j]:
                to_print += "x"
            else:
                to_print += "."
        to_print += "\n"
    if printing:
        print(to_print)
    return isTrack


def TrackDimensionReduction(track, xdim=3, ydim=2):
    # reduce the truth table of the track image to smaller matrix of floats (reduce the resolution)
    y = track.shape[0]
    x = track.shape[1]

    reduced_track = np.zeros((ydim, xdim))

    delta_x = round(x / xdim)
    delta_y = round(y / ydim)

    # Loop over all pixels ahead of the car
    for i in range(y):
        for j in range(x):
            reduced_track[i // delta_y][j // delta_x] += track[i][j] / (delta_x * delta_y)

    result=[]
    for i in range(ydim):
        for j in range(xdim):
            result.append(reduced_track[i][j])

    return result


def Sensors(track, num=7):
    y = track.shape[0]
    x = track.shape[1]
    angles = np.linspace(-np.pi/2, np.pi/2, num)
    results = []
    for a in angles:
        x_temp = round((x-1)/2)
        y_temp = y-1
        beam = 0.
        while track[y_temp][x_temp]:
            beam += 0.25
            y_temp = round(y-1 - beam*np.cos(a))
            x_temp = round(((x-1)/2) + beam*np.sin(a))
            if y_temp < 0 or x_temp > (x-1) or x_temp < 0:
                break
        results.append(beam/max(x,y))
    return results



def Telemetry(observation, printing=False):
    # return sensor data from the image
    R = ""
    G = ""
    B = ""
    velocity = 0.0
    RPM = np.zeros(4)
    gyro = 0.0
    wheel = 0.0
    for i in range(84, 96):
        for j in range(96):
            if observation[i][j][0] > 127:
                if i == 87:
                    if j > 20:
                        if j <= 71:
                            gyro -= 0.05
                        else:
                            gyro += 0.05
                R += "x"
            else:
                R += "."
            if observation[i][j][1] > 127:
                if i == 87 and j > 20:
                    if j <= 47:
                        wheel -= 0.1
                    else:
                        wheel += 0.1
                G += "x"
            else:
                G += "."
            if observation[i][j][2] > 127:
                if j == 12:
                    velocity += 1. / 5.
                if j == 17:
                    RPM[0] += 1. / 7.
                if j == 20:
                    RPM[1] += 1. / 7.
                if j == 22:
                    RPM[2] += 1. / 7.
                if j == 24:
                    RPM[3] += 1. / 7.

                B += "x"
            else:
                B += "."
        R += "\n"
        G += "\n"
        B += "\n"
    if printing:
        print(velocity)
        print(RPM)
        print(gyro)
        print(wheel)
    result = [velocity, wheel, gyro]
    result.extend(RPM)
    return result


def reformat_observation(observation):
    track = TrackMatrix(observation)
    TrackInput = TrackDimensionReduction(track)
    TelemetryInput = Telemetry(observation)
    SensorsInput = Sensors(track)

    input = TrackInput.copy()
    input.extend(TelemetryInput)
    input.extend(SensorsInput)
    return input


def action_discrete_to_continuous(action):
    if action == 0:
        action = [0,0,0]
    elif action == 1:
        action = [-1, 0, 0]
    elif action == 2:
        action = [1, 0, 0]
    elif action == 3:
        action = [0, 0.7, 0]
    elif action == 4:
        action = [0, 0, 0.8]
    else:
        print('error')
    return action

