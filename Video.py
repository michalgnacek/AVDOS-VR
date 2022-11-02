# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 21:06:47 2022

@author: Michal
"""

class Video:

    def __init__(self, events, name):
        self.events = events
        self.name = name
        self.arousal_ratings = []
        self.valence_ratings = []
        for event in events:
          if("Valence:" in event):
                self.arousal_ratings.append(int(event.split("Arousal:",1)[1][0]))
                self.valence_ratings.append(int(event.split("Valence:",1)[1][0]))
        self.average_arousal = self.__average_rating(self.arousal_ratings)
        self.average_valence = self.__average_rating(self.valence_ratings)
                
    def __average_rating(self, ratings_list):
        sum = 0
        if(ratings_list==[]):
            return 0
        else:
            for rating in ratings_list:
                sum = sum + rating
            return round(sum/len(ratings_list),3)
                    
    def __average_arousal(self):
        return  self.__average_rating(self.arousal_ratings)
    
    def __average_valence(self):
        return self.__average_rating(self.valence_ratings)
    

