'''
@file: search_factory.py
@breif: Facotry class for global planner.
@author: Winter
@update: 2023.3.2
'''
import sys, os

sys.path.append(f"{os.path.dirname(os.path.join(__file__))}/../..")

from global_planner import *

class SearchFactory(object):
    def __init__(self) -> None:
        pass

    def __call__(self, planner_name, **config):
        if planner_name == "a_star":
            return AStar(**config)
        elif planner_name == "dijkstra":
            return Dijkstra(**config)
        elif planner_name == "gbfs":
            return GBFS(**config)
        elif planner_name == "jps":
            return JPS(**config)
        elif planner_name == "d_star":
            return DStar(**config)
        elif planner_name == "lpa_star":
            return LPAStar(**config)
        elif planner_name == "d_star_lite":
            return DStarLite(**config)
        elif planner_name == "voronoi":
            return VoronoiPlanner(**config)
        elif planner_name == "theta_star":
            return ThetaStar(**config)
        elif planner_name == "lazy_theta_star":
            return LazyThetaStar(**config)
        elif planner_name == "rrt":
            return RRT(**config)
        elif planner_name == "rrt_connect":
            return RRTConnect(**config)
        elif planner_name == "rrt_star":
            return RRTStar(**config)
        elif planner_name == "informed_rrt":
            return InformedRRT(**config)
        elif planner_name == "aco":
            return ACO(**config)
        else:
            raise ValueError("The `planner_name` must be set correctly.")