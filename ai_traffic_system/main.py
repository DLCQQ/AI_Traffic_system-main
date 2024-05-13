import random
import time
import threading
import pygame
import sys
import os
import numpy as np


prediction_model_mode = True


import joblib


mj = joblib.load("./ai_traffic_system/model_joblib") 


def rl_ml_model_timer(flow):
    flow_percentile = np.array(flow).reshape(-1, 1)
    green_time_predict = np.ceil(mj.predict(flow_percentile))
    return green_time_predict.astype(int)


def ml_model_timer(flow):
    flow_percentile = np.array(flow).reshape(-1, 1)
    green_time_predict = np.ceil(mj.predict(flow_percentile))

    if green_time_predict < 3:
        green_time_predict = 2
    elif green_time_predict > 7:
        green_time_predict = 6
    return green_time_predict



defaultGreen = {0: 20, 1: 20, 2: 20, 3: 20}
defaultRed = 120
defaultYellow = 2

signals = []
noOfSignals = 4
currentGreen = 0  

nextGreen = (
    currentGreen + 1
) % noOfSignals  

currentYellow = 0  

speeds = {"car": 1.62}  

x = {
    "right": [0, 0, 0],
    "down": [542, 563, 637],
    "left": [1400, 1400, 1400],
    "up": [680, 723, 819],
}
y = {
    "right": [380, 410, 465],
    "down": [0, 0, 0],
    "left": [258, 315, 365],
    "up": [800, 800, 800],
}

vehicles = {
    "right": {0: [], 1: [], 2: [], "crossed": 0},
    "down": {0: [], 1: [], 2: [], "crossed": 0},
    "left": {0: [], 1: [], 2: [], "crossed": 0},
    "up": {0: [], 1: [], 2: [], "crossed": 0},
}
vehicleTypes = {0: "car"}


directionNumbers = {0: "right", 1: "down", 2: "left", 3: "up"}


signalCoods = [(400, 430), (690, 110), (970, 305), (690, 610)]
signalTimerCoods = [(450, 480), (760, 160), (1020, 355), (740, 660)]


stopLines = {"right": 350, "down": 197, "left": 1050, "up": 603}
defaultStop = {"right": 340, "down": 187, "left": 1060, "up": 610}


stoppingGap = 25  
movingGap = 30  



allowedVehicleTypes = {"car": True}


allowedVehicleTypesList = []

vehiclesTurned = {
    "right": {1: [], 2: []},
    "down": {1: [], 2: []},
    "left": {1: [], 2: []},
    "up": {1: [], 2: []},
}
vehiclesNotTurned = {
    "right": {1: [], 2: []},
    "down": {1: [], 2: []},
    "left": {1: [], 2: []},
    "up": {1: [], 2: []},
}

rotationAngle = 3 


mid = {
    "right": {"x": 560, "y": 465},
    "down": {"x": 560, "y": 310},
    "left": {"x": 860, "y": 310},
    "up": {"x": 815, "y": 495},
}

randomGreenSignalTimer = False

randomGreenSignalTimerRange = [10, 20]


timeElapsed = 0
simulationTime = 0  
timeElapsedCoods = (1050, 30)

vehicleCountTexts = ["0", "0", "0", "0"]
vehicleCountCoods = [(1050, 70), (1050, 110), (1050, 150), (1050, 190)]

count_Leg1 = 0
count_Leg2 = 0
count_Leg3 = 0
count_Leg4 = 0

total_flow_count = 1
totalflowcoods = (10, 110)

pygame.init()
simulation = pygame.sprite.Group()


class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.signalText = ""


class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, will_turn):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.willTurn = will_turn
        self.turned = 0
        self.rotateAngle = 0
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1
        self.crossedIndex = 0
        path = "ai_traffic_system/images/" + direction + "/" + vehicleClass + ".png"
        self.originalImage = pygame.image.load(path)
        self.image = pygame.image.load(path)

        if (
            len(vehicles[direction][lane]) > 1
            and vehicles[direction][lane][self.index - 1].crossed == 0
        ):
            if direction == "right":
                self.stop = vehicles[direction][lane][self.index - 1].stop
                -vehicles[direction][lane][self.index - 1].image.get_rect().width
                -stoppingGap
            elif direction == "left":
                self.stop = vehicles[direction][lane][self.index - 1].stop
                +vehicles[direction][lane][self.index - 1].image.get_rect().width
                +stoppingGap
            elif direction == "down":
                self.stop = vehicles[direction][lane][self.index - 1].stop
                -vehicles[direction][lane][self.index - 1].image.get_rect().height
                -stoppingGap
            elif direction == "up":
                self.stop = vehicles[direction][lane][self.index - 1].stop
                +vehicles[direction][lane][self.index - 1].image.get_rect().height
                +stoppingGap
        else:
            self.stop = defaultStop[direction]

        if direction == "right":
            temp = self.image.get_rect().width + stoppingGap
            x[direction][lane] -= temp
        elif direction == "left":
            temp = self.image.get_rect().width + stoppingGap
            x[direction][lane] += temp
        elif direction == "down":
            temp = self.image.get_rect().height + stoppingGap
            y[direction][lane] -= temp
        elif direction == "up":
            temp = self.image.get_rect().height + stoppingGap
            y[direction][lane] += temp
        simulation.add(self)

    def render(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def move(self):

        if self.direction == "right":
            if (
                self.crossed == 0
                and self.x + self.image.get_rect().width > stopLines[self.direction]
            ):
                self.crossed = 1
                vehicles[self.direction]["crossed"] += 1
                if self.willTurn == 0:
                    vehiclesNotTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = (
                        len(vehiclesNotTurned[self.direction][self.lane]) - 1
                    )
            if self.willTurn == 1:
                if self.lane == 1:
                    if (
                        self.crossed == 0
                        or self.x + self.image.get_rect().width
                        < stopLines[self.direction] + 365
                    ):
                        if (
                            self.x + self.image.get_rect().width <= self.stop
                            or (currentGreen == 0 and currentYellow == 0)
                            or self.crossed == 1
                        ) and (
                            self.index == 0
                            or self.x + self.image.get_rect().width
                            < (
                                vehicles[self.direction][self.lane][self.index - 1].x
                                - movingGap
                            )
                            or vehicles[self.direction][self.lane][
                                self.index - 1
                            ].turned
                            == 1
                        ):
                            self.x += self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(
                                self.originalImage, self.rotateAngle
                            )
                            self.x += 2.4  
                            self.y -= 2.8  
                            if self.rotateAngle == 90:
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = (
                                    len(vehiclesTurned[self.direction][self.lane]) - 1
                                )
                        else:
                            if self.crossedIndex == 0 or (
                                self.y
                                > (
                                    vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1
                                    ].y
                                    + vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1
                                    ]
                                    .image.get_rect()
                                    .height
                                    + movingGap
                                )
                            ):
                                self.y -= self.speed
                elif self.lane == 2:
                    if (
                        self.crossed == 0
                        or self.x + self.image.get_rect().width
                        < mid[self.direction]["x"]
                    ):
                        if (
                            self.x + self.image.get_rect().width <= self.stop
                            or (currentGreen == 0 and currentYellow == 0)
                            or self.crossed == 1
                        ) and (
                            self.index == 0
                            or self.x + self.image.get_rect().width
                            < (
                                vehicles[self.direction][self.lane][self.index - 1].x
                                - movingGap
                            )
                            or vehicles[self.direction][self.lane][
                                self.index - 1
                            ].turned
                            == 1
                        ):
                            self.x += self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(
                                self.originalImage, -self.rotateAngle
                            )
                            self.x += 2
                            self.y += 1.8
                            if self.rotateAngle == 90:
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                            self.crossedIndex = (
                                len(vehiclesTurned[self.direction][self.lane]) - 1
                            )
                        else:
                            if self.crossedIndex == 0 or (
                                (self.y + self.image.get_rect().height)
                                < (
                                    vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1
                                    ].y
                                    - movingGap
                                )
                            ):
                                self.y += self.speed
            else:
                if self.crossed == 0:
                    if (
                        self.x + self.image.get_rect().width <= self.stop
                        or (currentGreen == 0 and currentYellow == 0)
                    ) and (
                        self.index == 0
                        or self.x + self.image.get_rect().width
                        < (
                            vehicles[self.direction][self.lane][self.index - 1].x
                            - movingGap
                        )
                    ):
                        self.x += self.speed
                else:
                    if (self.crossedIndex == 0) or (
                        self.x + self.image.get_rect().width
                        < (
                            vehiclesNotTurned[self.direction][self.lane][
                                self.crossedIndex - 1
                            ].x
                            - movingGap
                        )
                    ):
                        self.x += self.speed
        elif self.direction == "down":
            if (
                self.crossed == 0
                and self.y + self.image.get_rect().height > stopLines[self.direction]
            ):
                self.crossed = 1
                vehicles[self.direction]["crossed"] += 1
                if self.willTurn == 0:
                    vehiclesNotTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = (
                        len(vehiclesNotTurned[self.direction][self.lane]) - 1
                    )
            if self.willTurn == 1:
                if self.lane == 2:
                    if (
                        self.crossed == 0
                        or self.y + self.image.get_rect().height
                        < stopLines[self.direction] + 210
                    ):
                        if (
                            self.y + self.image.get_rect().height <= self.stop
                            or (currentGreen == 1 and currentYellow == 0)
                            or self.crossed == 1
                        ) and (
                            self.index == 0
                            or self.y + self.image.get_rect().height
                            < (
                                vehicles[self.direction][self.lane][self.index - 1].y
                                - movingGap
                            )
                            or vehicles[self.direction][self.lane][
                                self.index - 1
                            ].turned
                            == 1
                        ):
                            self.y += self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(
                                self.originalImage, self.rotateAngle
                            )
                            self.x += 1.2
                            self.y += 1.8
                            if self.rotateAngle == 90:
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = (
                                    len(vehiclesTurned[self.direction][self.lane]) - 1
                                )
                        else:
                            if self.crossedIndex == 0 or (
                                (self.x + self.image.get_rect().width)
                                < (
                                    vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1
                                    ].x
                                    - movingGap
                                )
                            ):
                                self.x += self.speed
                elif self.lane == 1:
                    if (
                        self.crossed == 0
                        or self.y + self.image.get_rect().height
                        < mid[self.direction]["y"]
                    ):
                        if (
                            self.y + self.image.get_rect().height <= self.stop
                            or (currentGreen == 1 and currentYellow == 0)
                            or self.crossed == 1
                        ) and (
                            self.index == 0
                            or self.y + self.image.get_rect().height
                            < (
                                vehicles[self.direction][self.lane][self.index - 1].y
                                - movingGap
                            )
                            or vehicles[self.direction][self.lane][
                                self.index - 1
                            ].turned
                            == 1
                        ):
                            self.y += self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(
                                self.originalImage, -self.rotateAngle
                            )
                            self.x -= 2.5
                            self.y += 2
                            if self.rotateAngle == 90:
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = (
                                    len(vehiclesTurned[self.direction][self.lane]) - 1
                                )
                        else:
                            if self.crossedIndex == 0 or (
                                self.x
                                > (
                                    vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1
                                    ].x
                                    + vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1
                                    ]
                                    .image.get_rect()
                                    .width
                                    + movingGap
                                )
                            ):
                                self.x -= self.speed
            else:
                if self.crossed == 0:
                    if (
                        self.y + self.image.get_rect().height <= self.stop
                        or (currentGreen == 1 and currentYellow == 0)
                    ) and (
                        self.index == 0
                        or self.y + self.image.get_rect().height
                        < (
                            vehicles[self.direction][self.lane][self.index - 1].y
                            - movingGap
                        )
                    ):
                        self.y += self.speed
                else:
                    if (self.crossedIndex == 0) or (
                        self.y + self.image.get_rect().height
                        < (
                            vehiclesNotTurned[self.direction][self.lane][
                                self.crossedIndex - 1
                            ].y
                            - movingGap
                        )
                    ):
                        self.y += self.speed
        elif self.direction == "left":
            if self.crossed == 0 and self.x < stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]["crossed"] += 1
                if self.willTurn == 0:
                    vehiclesNotTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = (
                        len(vehiclesNotTurned[self.direction][self.lane]) - 1
                    )
            if self.willTurn == 1:
                if self.lane == 2:
                    if self.crossed == 0 or self.x > stopLines[self.direction] - 440:
                        if (
                            self.x >= self.stop
                            or (currentGreen == 2 and currentYellow == 0)
                            or self.crossed == 1
                        ) and (
                            self.index == 0
                            or self.x
                            > (
                                vehicles[self.direction][self.lane][self.index - 1].x
                                + vehicles[self.direction][self.lane][self.index - 1]
                                .image.get_rect()
                                .width
                                + movingGap
                            )
                            or vehicles[self.direction][self.lane][
                                self.index - 1
                            ].turned
                            == 1
                        ):
                            self.x -= self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(
                                self.originalImage, self.rotateAngle
                            )
                            self.x -= 1
                            self.y += 1.2
                            if self.rotateAngle == 90:
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = (
                                    len(vehiclesTurned[self.direction][self.lane]) - 1
                                )
                        else:
                            if self.crossedIndex == 0 or (
                                (self.y + self.image.get_rect().height)
                                < (
                                    vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1
                                    ].y
                                    - movingGap
                                )
                            ):
                                self.y += self.speed
                elif self.lane == 1:
                    if self.crossed == 0 or self.x > mid[self.direction]["x"]:
                        if (
                            self.x >= self.stop
                            or (currentGreen == 2 and currentYellow == 0)
                            or self.crossed == 1
                        ) and (
                            self.index == 0
                            or self.x
                            > (
                                vehicles[self.direction][self.lane][self.index - 1].x
                                + vehicles[self.direction][self.lane][self.index - 1]
                                .image.get_rect()
                                .width
                                + movingGap
                            )
                            or vehicles[self.direction][self.lane][
                                self.index - 1
                            ].turned
                            == 1
                        ):
                            self.x -= self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(
                                self.originalImage, -self.rotateAngle
                            )
                            self.x -= 1.8
                            self.y -= 2.5
                            if self.rotateAngle == 90:
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = (
                                    len(vehiclesTurned[self.direction][self.lane]) - 1
                                )
                        else:
                            if self.crossedIndex == 0 or (
                                self.y
                                > (
                                    vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1
                                    ].y
                                    + vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1
                                    ]
                                    .image.get_rect()
                                    .height
                                    + movingGap
                                )
                            ):
                                self.y -= self.speed
            else:
                if self.crossed == 0:
                    if (
                        self.x >= self.stop
                        or (currentGreen == 2 and currentYellow == 0)
                    ) and (
                        self.index == 0
                        or self.x
                        > (
                            vehicles[self.direction][self.lane][self.index - 1].x
                            + vehicles[self.direction][self.lane][self.index - 1]
                            .image.get_rect()
                            .width
                            + movingGap
                        )
                    ):
                        self.x -= self.speed
                else:
                    if (self.crossedIndex == 0) or (
                        self.x
                        > (
                            vehiclesNotTurned[self.direction][self.lane][
                                self.crossedIndex - 1
                            ].x
                            + vehiclesNotTurned[self.direction][self.lane][
                                self.crossedIndex - 1
                            ]
                            .image.get_rect()
                            .width
                            + movingGap
                        )
                    ):
                        self.x -= self.speed
        elif self.direction == "up":
            if self.crossed == 0 and self.y < stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]["crossed"] += 1
                if self.willTurn == 0:
                    vehiclesNotTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = (
                        len(vehiclesNotTurned[self.direction][self.lane]) - 1
                    )
            if self.willTurn == 1:
                if self.lane == 1:
                    if self.crossed == 0 or self.y > stopLines[self.direction] - 200:
                        if (
                            self.y >= self.stop
                            or (currentGreen == 3 and currentYellow == 0)
                            or self.crossed == 1
                        ) and (
                            self.index == 0
                            or self.y
                            > (
                                vehicles[self.direction][self.lane][self.index - 1].y
                                + vehicles[self.direction][self.lane][self.index - 1]
                                .image.get_rect()
                                .height
                                + movingGap
                            )
                            or vehicles[self.direction][self.lane][
                                self.index - 1
                            ].turned
                            == 1
                        ):
                            self.y -= self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(
                                self.originalImage, self.rotateAngle
                            )
                            self.x -= 2
                            self.y -= 1.2
                            if self.rotateAngle == 90:
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = (
                                    len(vehiclesTurned[self.direction][self.lane]) - 1
                                )
                        else:
                            if self.crossedIndex == 0 or (
                                self.x
                                > (
                                    vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1
                                    ].x
                                    + vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1
                                    ]
                                    .image.get_rect()
                                    .width
                                    + movingGap
                                )
                            ):
                                self.x -= self.speed
                elif self.lane == 2:
                    if self.crossed == 0 or self.y > mid[self.direction]["y"]:
                        if (
                            self.y >= self.stop
                            or (currentGreen == 3 and currentYellow == 0)
                            or self.crossed == 1
                        ) and (
                            self.index == 0
                            or self.y
                            > (
                                vehicles[self.direction][self.lane][self.index - 1].y
                                + vehicles[self.direction][self.lane][self.index - 1]
                                .image.get_rect()
                                .height
                                + movingGap
                            )
                            or vehicles[self.direction][self.lane][
                                self.index - 1
                            ].turned
                            == 1
                        ):
                            self.y -= self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(
                                self.originalImage, -self.rotateAngle
                            )
                            self.x += 1
                            self.y -= 1
                            if self.rotateAngle == 90:
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = (
                                    len(vehiclesTurned[self.direction][self.lane]) - 1
                                )
                        else:
                            if self.crossedIndex == 0 or (
                                self.x
                                < (
                                    vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1
                                    ].x
                                    - vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1
                                    ]
                                    .image.get_rect()
                                    .width
                                    - movingGap
                                )
                            ):
                                self.x += self.speed
            else:
                if self.crossed == 0:
                    if (
                        self.y >= self.stop
                        or (currentGreen == 3 and currentYellow == 0)
                    ) and (
                        self.index == 0
                        or self.y
                        > (
                            vehicles[self.direction][self.lane][self.index - 1].y
                            + vehicles[self.direction][self.lane][self.index - 1]
                            .image.get_rect()
                            .height
                            + movingGap
                        )
                    ):
                        self.y -= self.speed
                else:
                    if (self.crossedIndex == 0) or (
                        self.y
                        > (
                            vehiclesNotTurned[self.direction][self.lane][
                                self.crossedIndex - 1
                            ].y
                            + vehiclesNotTurned[self.direction][self.lane][
                                self.crossedIndex - 1
                            ]
                            .image.get_rect()
                            .height
                            + movingGap
                        )
                    ):
                        self.y -= self.speed



def initialize():

    if prediction_model_mode:


        L1_percen = count_Leg1 / total_flow_count
        L2_percen = count_Leg2 / total_flow_count
        L3_percen = count_Leg3 / total_flow_count
        L4_percen = count_Leg4 / total_flow_count

        T1_predict = ml_model_timer(L1_percen)
        T2_predict = ml_model_timer(L2_percen)
        T3_predict = ml_model_timer(L3_percen)
        T4_predict = ml_model_timer(L4_percen)
        ts1 = TrafficSignal(0, defaultYellow, T1_predict)
        signals.append(ts1)
        tts1 = ts1.red + ts1.yellow + ts1.green
        ts2 = TrafficSignal(tts1, defaultYellow, T2_predict)
        signals.append(ts2)
        tts2 = tts1 + ts2.yellow + ts2.green
        ts3 = TrafficSignal(tts2, defaultYellow, T3_predict)
        signals.append(ts3)
        tts3 = tts2 + ts3.yellow + ts3.green
        ts4 = TrafficSignal(tts3, defaultYellow, T4_predict)
        signals.append(ts4)

    else:
        ts1 = TrafficSignal(0, defaultYellow, defaultGreen[0])
        signals.append(ts1)
        ts2 = TrafficSignal(ts1.yellow + ts1.green, defaultYellow, defaultGreen[1])
        signals.append(ts2)
        ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[2])
        signals.append(ts3)
        ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[3])
        signals.append(ts4)
    repeat()



def printStatus():
    for i in range(0, 4):
        if signals[i] != None:
            if i == currentGreen:
                if currentYellow == 0:
                    print(
                        " GREEN TS",
                        i + 1,
                        "-> r:",
                        signals[i].red,
                        "-> y:",
                        signals[i].yellow,
                        "-> g:",
                        signals[i].green,
                    )
                else:
                    print(
                        "YELLOW TS",
                        i + 1,
                        "-> r:",
                        signals[i].red,
                        "-> y:",
                        signals[i].yellow,
                        "-> g:",
                        signals[i].green,
                    )
            else:
                print(
                    "   RED TS",
                    i + 1,
                    "-> r:",
                    signals[i].red,
                    "-> y:",
                    signals[i].yellow,
                    "-> g:",
                    signals[i].green,
                )
    print()


def repeat():
    global currentGreen, currentYellow, nextGreen
    while (
        signals[currentGreen].green > 0
    ):  
        printStatus()
        updateValues()
        time.sleep(
            1
        ) 
    currentYellow = 1  
    
    for i in range(0, 3):
        for vehicle in vehicles[directionNumbers[currentGreen]][i]:
            vehicle.stop = defaultStop[directionNumbers[currentGreen]]
    while (
        signals[currentGreen].yellow > 0
    ):  
        printStatus()
        updateValues()
        time.sleep(
            1
        )  
    currentYellow = 0  

    if prediction_model_mode:
        signals[0].green = ml_model_timer(count_Leg1 / total_flow_count)
        signals[1].green = ml_model_timer(count_Leg2 / total_flow_count)
        signals[2].green = ml_model_timer(count_Leg3 / total_flow_count)
        signals[3].green = ml_model_timer(count_Leg4 / total_flow_count)
    else:
        signals[currentGreen].green = defaultGreen[currentGreen]
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed

    currentGreen = nextGreen  
    nextGreen = (currentGreen + 1) % noOfSignals  
    signals[nextGreen].red = (
        signals[currentGreen].yellow + signals[currentGreen].green
    )  
    repeat()



def updateValues():
    for i in range(0, noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                signals[i].green -= 1
            else:
                signals[i].yellow -= 1
        else:
            signals[i].red -= 1


def generateVehicles():
    global total_flow_count, count_Leg1, count_Leg2, count_Leg3, count_Leg4
    while True:
        vehicle_type = random.choice(allowedVehicleTypesList)
        lane_number = random.randint(1, 2)
        will_turn = 0

        if lane_number == 1:
            temp = random.randint(0, 99)
            if temp < 40:
                will_turn = 1
        elif lane_number == 2:
            temp = random.randint(0, 99)
            if temp < 40:
                will_turn = 1
        temp = random.randint(0, 100)

        direction_number = 0

        dist = [5, 11, 56, 101]
        if temp < dist[0]:
            direction_number = 1  
            count_Leg2 += 1
        elif temp < dist[1]:
            direction_number = 3 
            count_Leg4 += 1
        elif temp < dist[2]:
            direction_number = 0 
            count_Leg1 += 1
        elif temp < dist[3]:
            direction_number = 2  
            count_Leg3 += 1
        Vehicle(
            lane_number,
            vehicleTypes[vehicle_type],
            direction_number,
            directionNumbers[direction_number],
            will_turn,
        )

        time.sleep(
            1.25
        ) 
        total_flow_count += 1
        print("Total flow count: ", total_flow_count)



def showStats():
    totalVehicles = 0
    print("Direction-wise Vehicle crossed Counts of Lanes#")
    for i in range(0, 4):
        if signals[i] != None:
            print("Direction", i + 1, ":", vehicles[directionNumbers[i]]["crossed"])
            totalVehicles += vehicles[directionNumbers[i]]["crossed"]
    print("Total vehicles passed: ", totalVehicles)
    print("Total time: ", timeElapsed)


def simTime():

    global timeElapsed, simulationTime
    while True:
        timeElapsed += 1
        time.sleep(1)
        if timeElapsed == simulationTime:
            showStats()
            os._exit(1)


class Main:
    global allowedVehicleTypesList
    i = 0
    for vehicleType in allowedVehicleTypes:
        if allowedVehicleTypes[vehicleType]:
            allowedVehicleTypesList.append(i)
        i += 1
    thread1 = threading.Thread(
        name="initialization", target=initialize, args=()
    )  
    thread1.daemon = True
    thread1.start()

  
    black = (0, 0, 0)
    white = (255, 255, 255)

 
    screenWidth = 1400
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)

   
    background = pygame.image.load("ai_traffic_system/images/intersection4.png")

    screen = pygame.display.set_mode(screenSize)
    pygame.display.set_caption("SIMULATION")


    redSignal = pygame.image.load("ai_traffic_system/images/signals/red.png")
    yellowSignal = pygame.image.load("ai_traffic_system/images/signals/yellow.png")
    greenSignal = pygame.image.load("ai_traffic_system/images/signals/green.png")
    font = pygame.font.Font(None, 28)
    thread2 = threading.Thread(
        name="generateVehicles1", target=generateVehicles, args=()
    )  
    thread2.daemon = True
    thread2.start()

    thread5 = threading.Thread(name="simTime", target=simTime, args=())
    thread5.daemon = True
    thread5.start()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                showStats()
                sys.exit()

        screen.blit(background, (0, 0))  
        for i in range(
            0, noOfSignals
        ): 
            if i == currentGreen:
                if currentYellow == 1:
                    signals[i].signalText = signals[i].yellow
                    screen.blit(yellowSignal, signalCoods[i])
                else:
                    signals[i].signalText = signals[i].green
                    screen.blit(greenSignal, signalCoods[i])
            else:
                signals[i].signalText = signals[i].red
                screen.blit(redSignal, signalCoods[i])
                if signals[i].red <= 200:
                    signals[i].signalText = signals[i].red
                else:
                    signals[i].signalText = "---"
                screen.blit(redSignal, signalCoods[i])
        signalTexts = ["", "", "", ""]

 
        for i in range(0, noOfSignals):
            signalTexts[i] = font.render(str(signals[i].signalText), True, white, black)
            screen.blit(signalTexts[i], signalTimerCoods[i])

        
        for vehicle in simulation:
            screen.blit(vehicle.image, [vehicle.x, vehicle.y])
            vehicle.move()

      
        speed_limit = font.render("Скоростной лимит 50 км/ч", True, black, white)
        car_speed_text = font.render(
            "Скорость машин 41.2 км/ч", True, black, white
        )
        screen.blit(speed_limit, (10, 30))
        screen.blit(car_speed_text, (10, 70))

        
        for i in range(0, noOfSignals):
            displayText = vehicles[directionNumbers[i]]["crossed"]
            vehicleCountTexts[i] = font.render(
                "Дорога #" + str(i + 1) + " Машин пройденно : " + str(displayText),
                True,
                black,
                white,
            )
            screen.blit(vehicleCountTexts[i], vehicleCountCoods[i])

       
        timeElapsedText = font.render(
            ("Таймер: " + str(timeElapsed)), True, black, white
        )
        screen.blit(timeElapsedText, timeElapsedCoods)

        
        flowcount = font.render(
            "Машин урегулированно: " + str(total_flow_count), True, black, white
        )
        screen.blit(flowcount, totalflowcoods)

        pygame.display.update()




if __name__ == "__main__":
    Main()
