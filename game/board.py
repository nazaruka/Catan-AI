import enum
from collections import defaultdict
from collections import namedtuple
from itertools import chain
from operator import itemgetter
from pprint import pformat
from textwrap import wrap
from typing import List, Tuple, Set, Dict

import networkx
import pygraphviz
import numpy as np

from algorithms.tree_diameter import tree_diameter
from game.pieces import Colony, Road
from game.resource import Resource

"""
Structure
---------
The board is represented as follows:
It consists of two parts:
a graph that represents the items around the hexagon, and an array of the hexagons.
The graph will hold the "shape":
 -each vertex will be a place a house can be built in
 -each edge will be a place a road can be paved at
THe array will hold the "data":
 -each item will be a hexagon, that consists of:
    --the element (Wheat, Metal, Clay or Wood)
    --the number (2-12)
    --Is there a robber on the hexagon or not
each edge & vertex in the graph will be bi-directionally linked to it's hexagons, for easy traversal

Example
-------
This map (W2 means wool  with the number 5 on it, L2 is lumber with 2 on it):

    O     O
 /    \ /    \
O      O      O
| (W5) | (L2) |
O      O      O
 \    / \    /
    O     O

In the DS, will be represented as follows:
The array:
 ---- ----
| W5 | L2 |
 ---- ----
The graph will have the shape of the map, where the edges are \,/,|
and the vertices are O.
    O     O
 /    \ /    \
O      O      O
|      |      |
O      O      O
 \    / \    /
    O     O

"""


@enum.unique
class Harbor(enum.Enum):
    """
    Harbor types. Harbors are locations one can exchange resources in.
    specific harbors at 1:2 ratio (1 specific harbor per Resource type)
    generic at 1:3 ratio (4 generic harbors)
    note that the enum numbers correspond to the Resource enum, for easy mapping between the two
    """
    HarborBrick = Resource.Brick.value
    HarborLumber = Resource.Lumber.value
    HarborWool = Resource.Wool.value
    HarborGrain = Resource.Grain.value
    HarborOre = Resource.Ore.value
    HarborGeneric = 5


Location = int
"""
Location is a vertex in the graph
A place that can be colonised (with a settlement, and later with a city)
"""

Path = Tuple[Location, Location]
"""
Path is an edge in the graph
A place that a road can be paved in
"""


class Land(namedtuple('LandTuple', ['resource', 'dice_value', 'identifier', 'locations', 'colonies'])):
    """
    Land is an element in the lands array
    A hexagon in the catan map, that has (in this order):
     -a resource type
     -a number between [2,12]
     -an id
     -Locations list (the locations around it)
     -all adjacent colonies
    """

    def __deepcopy__(self, memo_dict=None):
        return self


def path_key(edge):
    return min(edge) * 100 + max(edge)


class Board:
    player = 'p'
    lands = 'l'

    def __init__(self, seed: int = None):
        """
        Board of the game settlers of catan
        :param seed: optional parameter. send the same number in the range [0,1) to get the same map
        """
        assert seed is None or (isinstance(seed, int) and seed > 0)

        self._shuffle = np.random.RandomState(seed).shuffle
        self._player_colonies_points = defaultdict(int)
        self._players_by_roads = {}

        self._create_and_shuffle_lands()
        self._create_graph()
        self._set_attributes()
        self._create_harbors()

    def get_settleable_locations_by_player(self, player) -> List[Location]:
        """
        get non-colonised (empty vertices) locations on map that this player can settle
        :param player: the player to get settleable location by
        :return: list of locations on map that the player can settle locations on
        """
        colonised_by_player_count = 0
        non_colonised = []
        for v in self._roads_and_colonies.nodes():
            if self.is_colonised_by(player, v):
                colonised_by_player_count += 1
            elif not self.is_colonised(v):
                non_colonised.append(v)

        if colonised_by_player_count < 2:
            def condition_on_path(path):
                return True
        else:
            def condition_on_path(path):
                return self.has_road_been_paved_by(player, path)

        coloniseable = []
        # make sure there's no settlement one-hop from settleable locations
        # and that there's an edge from that location, u-v
        # the 2nd condition (edge u-v exists) is checked only if it isn't the first 2 settlements
        for u in non_colonised:
            is_coloniseable = True
            one_hop_from_non_colonised = []
            for v in self._roads_and_colonies.neighbors(u):
                if self.is_colonised(v):
                    is_coloniseable = False
                    break
                if condition_on_path((u, v)):
                    one_hop_from_non_colonised.append(v)
            if not is_coloniseable:
                continue

            if colonised_by_player_count >= 2:
                # make sure there's another edge from u
                # that is, there's a road u-v-w paved by this player
                # this is checked only if it isn't the first 2 settlements
                is_coloniseable = False
                for v in one_hop_from_non_colonised:
                    for w in self._roads_and_colonies.neighbors(v):
                        if w != u and condition_on_path((u, v)):
                            is_coloniseable = True
                            break

            if is_coloniseable:
                coloniseable.append(u)
        return coloniseable

    def get_settlements_by_player(self, player) -> List[Location]:
        """
        get player's settlements on map that this player can settle with a city
        unlike get_locations_colonised_by_player, this method returns only the
        settlements' locations. It doesn't return cities locations
        :param player: the player to get settlements
        :return: list of locations on map that the player can settle a city on
        """
        return [v for v in self._roads_and_colonies.nodes()
                if self._roads_and_colonies.nodes[v][Board.player] == (player, Colony.Settlement)]

    def get_locations_colonised_by_player(self, player) -> List[Location]:
        """
        get the colonies owned by given player
        unlike get_settlements_by_player, this method returns both the
        settlements' and the cities' locations
        :param player: the player to get the colonies of
        :return: list of locations that have colonies of the given player
        """
        return [v for v in self._roads_and_colonies.nodes() if self.is_colonised_by(player, v)]

    def get_roads_paved_by_player(self, player):
        """
        get all the roads the player paved
        :param player: player of which the paths
        :return: List[Path]
        """
        return [e for e in self._roads_and_colonies.edges() if self.has_road_been_paved_by(player, e)]

    def get_unpaved_paths_near_player(self, player) -> List[Path]:
        """
        get unpaved (empty edges) paths on map that this player can pave
        :param player: the player to get paths on map that he can pave
        :return: list of paths the player can pave a road in
        """
        roads = [e for e in self._roads_and_colonies.edges() if self.has_road_been_paved_by(player, e)]

        less_than_two_roads_paved = len(roads) < 2
        if less_than_two_roads_paved:
            return [(max(u, v), min(u, v)) for u in self.get_settlements_by_player(player)
                    for v in self._roads_and_colonies.neighbors(u)
                    if self.has_road_been_paved_by(None, (u, v))]

        uncolonised_by_other_players = [v for v in set(chain(*roads))
                                        if self.is_colonised_by(player, v) or not self.is_colonised(v)]
        return [(max(u, v), min(u, v)) for u in uncolonised_by_other_players
                for v in self._roads_and_colonies.neighbors(u)
                if self.has_road_been_paved_by(None, (u, v))]

    def get_surrounding_resources(self, location: Location) -> List[Resource]:
        """
        get resources surrounding the settlement in this location
        :param location: the location to get the resources around
        :return: list of resources
        """
        return [land.resource for land in self._roads_and_colonies.nodes[location][Board.lands]
                if land.resource is not None]

    def get_surrounding_dice_values(self, location: Location) -> List[int]:
        """
        get numbers surrounding the settlement in this location
        :param location: the location to get the numbers around
        :return: list of numbers
        """
        return [land.dice_value for land in self._roads_and_colonies.nodes[location][Board.lands]
                if land.resource is not None]

    def get_adjacent_to_path_dice_values(self, path: Path):
        """
        get numbers adjacent to this path
        :param path: the path to get the numbers around
        :return: list of numbers
        """
        return [land.dice_value for land in self._roads_and_colonies[path[0]][path[1]][Board.lands]
                if land.resource is not None]

    def get_colonies_score(self, player) -> int:
        """
        get the colonies score-count of a single player
        that is the sum of points the player got for his colonies
        :param player: a player to get the colonies-score of
        :return: int, the score of the specified player
        """
        return self._player_colonies_points[player]

    def get_longest_road_length_of_player(self, player) -> int:
        """
        get the longest road length of specified player.
        NOTE: if player has less than 5 roads in total, it returns 4
        that's because it means he can't have the "longest-road" card anyways,
        so computing the longest road is unnecessary
        :param player: the player fir whom the longest road is calculated
        :return: max(4, the length of the longest road of specified player)
        """
        roads_paved_by_player = [e for e in self._roads_and_colonies.edges()
                                 if self.has_road_been_paved_by(player, e)]

        roads_threshold = 4
        if len(roads_paved_by_player) <= roads_threshold:
            return roads_threshold
        sub_graph_of_player = networkx.Graph(roads_paved_by_player)
        max_road_length = roads_threshold

        connected_components_and_edge_count_sorted_by_edge_count = sorted(
            ((g, g.size()) for g in networkx.connected_component_subgraphs(sub_graph_of_player, copy=False)),
            key=itemgetter(1), reverse=True)

        for g, edges_count in connected_components_and_edge_count_sorted_by_edge_count:
            if edges_count <= max_road_length:
                return max_road_length
            if networkx.is_tree(g):
                max_road_length = max(max_road_length, tree_diameter(g) - 1)
            else:
                for w in g.nodes():
                    max_road_length = max(max_road_length, Board._compute_longest_road_length(g, w, set()))
                    if max_road_length == edges_count:
                        return max_road_length
        return max_road_length

    def get_players_to_resources_by_dice_value(self, dice_value: int) -> Dict:
        """
        get the resources that players get when the dice roll specified number
        :param dice_value: the number the dice rolled
        :return: Dict[player, Dict[Resource, int]], a dictionary of plaers to
        the resources they should receive
        """
        assert 2 <= dice_value <= 12 and dice_value != 7
        lands_with_this_number = [land for land in self._lands
                                  if land.dice_value == dice_value and self._robber_land != land]
        players_to_resources = {player: {resource: 0 for resource in Resource}
                                for player in self._player_colonies_points.keys()}
        for land in lands_with_this_number:
            for location in land.locations:
                if self.is_colonised(location):
                    player = self._roads_and_colonies.nodes[location][Board.player][0]
                    colony = self.get_colony_type_at_location(location)
                    players_to_resources[player][land.resource] += colony.value
        return players_to_resources

    def get_colony_type_at_location(self, location: Location) -> Colony:
        return self._roads_and_colonies.nodes[location][Board.player][1]

    def set_location(self, player, location: Location, colony: Colony):
        """
        settle/unsettle given colony type in given location by given player
        :param player: the player to settle/unsettle a settlement of
        :param location: the location to put the settlement on
        :param colony: the colony type to put (settlement/city)
        :return: None
        """
        assert not (player is None and colony != Colony.Uncolonised)

        vertex_attributes = self._roads_and_colonies.nodes[location]

        previous_colony = self.get_colony_type_at_location(location)
        self._player_colonies_points[player] -= previous_colony.value
        self._player_colonies_points[player] += colony.value

        if colony is colony.Uncolonised and previous_colony is not colony.Uncolonised:
            for land in vertex_attributes[Board.lands]:
                land.colonies.pop()
        elif colony is not colony.Uncolonised and previous_colony is colony.Uncolonised:
            for land in vertex_attributes[Board.lands]:
                land.colonies.append(colony)

        if colony == Colony.Uncolonised:
            player = None
        vertex_attributes[Board.player] = (player, colony)

        if __debug__:
            sum_of_settlements_and_cities_points = 0
            for v in self._roads_and_colonies.nodes():
                sum_of_settlements_and_cities_points += self.get_colony_type_at_location(v).value

            sum_of_points = 0
            for points in self._player_colonies_points.values():
                sum_of_points += points

            assert sum_of_points == sum_of_settlements_and_cities_points

    def set_path(self, player, path: Path, road: Road):
        """
        pave/un-pave road in given location by given player
        NOTE that if the road type is Road.Unpaved, then the player is irrelevant
        :param player: the player that paves/un-paves the road
        :param path: the path on the map to pave/un-pave the road at
        :param road: road type. Road.Paved to pave, Road.Unpaved to un-pave
        :return: None
        """
        assert not (player is None and road != Road.Unpaved)
        if road == Road.Unpaved:
            player = None
        self._roads_and_colonies[path[0]][path[1]][Board.player] = (player, road)
        self._players_by_roads[path_key(path)] = player

    def get_robber_land(self) -> Land:
        """
        get the land where the robber currently lays
        :return: Land, where the robber is located
        """
        return self._robber_land

    def set_robber_land(self, land: Land):
        """
        set the land where the robber lays
        :param land: the land where the robber will be located
        :return: None
        """
        self._robber_land = land

    def is_colonised(self, location: Location) -> bool:
        """
        indicate whether the specified location is colonised
        :param location: the location to check
        :return: True if specified location is colonised, false otherwise
        """
        return not self.is_colonised_by(None, location)

    def is_colonised_by(self, player, location: Location) -> bool:
        """
        indicate whether the specified location is colonised by specified player
        :param location: the location to check
        :param player: the player to check
        :return: True if specified location is colonised by player, false otherwise
        """
        return self._roads_and_colonies.nodes[location][Board.player][0] is player

    def has_road_been_paved_by(self, player, path: Path):
        """
        indicate whether a road has been paved in specified location by specified player
        :param player: the player to check if he paved a road in that path
        :param path: the path to check if the player paved a road at
        :return: True if road on that path has been paved by given player, False otherwise
        """
        return self._players_by_roads[path_key(path)] is player

    def plot_map(self, file_name='tmp.png', dice=None):
        vertices_by_players = self.get_locations_by_players()
        edges_by_players = self.get_paths_by_players()

        g = networkx.nx_agraph.to_agraph(self._roads_and_colonies)
        colors = ['orange', 'darkgreen', 'blue', 'red']
        for player in vertices_by_players.keys():
            color = 'grey'
            if player is not None:
                color = colors.pop()
            for vertex in vertices_by_players[player]:
                g.get_node(vertex).attr['color'] = color
                g.get_node(vertex).attr['fillcolor'] = color
                g.get_node(vertex).attr['style'] = 'filled'
                g.get_node(vertex).attr['fontsize'] = 25
                g.get_node(vertex).attr['fontname'] = 'times-bold'
                if self.get_colony_type_at_location(vertex) == Colony.City:
                    g.get_node(vertex).attr['shape'] = 'doublecircle'
                else:
                    g.get_node(vertex).attr['shape'] = 'circle'
                g.get_node(vertex).attr['penwidth'] = 2
            for u, v in edges_by_players[player]:
                g.get_edge(u, v).attr['color'] = color
                g.get_edge(u, v).attr['penwidth'] = 5
        for land in self._lands:
            land_node_id = 'land ' + str(land.identifier)
            g.add_node(land_node_id)
            resource = 'desert' if land.resource is None else land.resource.name
            robber = '@' if self._robber_land.identifier == land.identifier else ''
            land_label = resource + '\n' + robber + '\n' + str(land.dice_value)
            land_node = g.get_node(land_node_id)
            land_node.attr['label'] = land_label
            land_node.attr['fontsize'] = 20
            land_node.attr['fontname'] = 'times-bold'
            land_node.attr['shape'] = 'hexagon'
            land_node.attr['color'] = 'transparent'
            for node in land.locations:
                g.add_edge(node, land_node)
                g.get_edge(node, land_node).attr['color'] = 'transparent'

        blocks = []
        for v in self._player_colonies_points.keys():
            blocks.append('\\n'.join(wrap(pformat(
                {k: v for k, v in v.__dict__.items()
                 if k not in {'_random_choice', 'expectimax_alpha_beta', '_timeout_seconds', '_players_and_factors',
                              'weights'}}), width=45)).replace('{', '').replace('}', '').replace(',', '')
                          .replace('DevelopmentCard.', '').replace('<', '').replace('>', '').replace("'", '')
                          .replace('Knight: 0', 'Knight').replace('RoadBuilding: 2', 'Road Building')
                          .replace('VictoryPoint: 1', 'Victory Point').replace('Monopoly: 3', 'Monopoly')
                          .replace('YearOfPlenty: 4', 'Year Of Plenty').replace('Resource.', '').replace('Colony.', '')
                          .replace('.Paved', '').replace('Road: 1', 'Road').replace(': 0:', ':').replace(': 1:', ':')
                          .replace(': 2:', ':').replace(': 3:', ':').replace(': 4:', ':')
                          .replace('pieces:', 'pieces:\\n').replace('resources', '\\nresources'))
        blocks.append('rolled:\\n{}'.format(dice))
        g.add_node('game_data', shape='record', label='|'.join(blocks), fontsize=20, fontname='times-bold')
        g.add_edge('game_data', 26)
        g.get_edge('game_data', 26).attr['len'] = 8
        g.get_edge('game_data', 26).attr['color'] = 'transparent'
        g.layout()
        g.draw(file_name)

    def get_paths_by_players(self):
        """
        get players to paths dictionary
        my_board.get_paths_by_players()[None] == all the unpaved paths
        :return: Dict[Player, List[Location]]
        """
        edges_by_players = {player: self.get_roads_paved_by_player(player)
                            for player in self._player_colonies_points.keys()}
        edges_by_players[None] = [e for e in self._roads_and_colonies.edges()
                                  if self.has_road_been_paved_by(None, e)]
        return edges_by_players

    def get_locations_by_players(self):
        """
        get players to locations dictionary
        my_board.get_locations_by_players()[None] == all the non-colonised locations
        :return: Dict[Player, List[Location]]
        """
        vertices_by_players = {
            player: [v for v in self._roads_and_colonies.nodes() if self.is_colonised_by(player, v)]
            for player in self._player_colonies_points.keys()
            }
        vertices_by_players[None] = [v for v in self._roads_and_colonies.nodes() if not self.is_colonised(v)]
        return vertices_by_players

    def is_player_on_harbor(self, player, harbor: Harbor) -> bool:
        """
        indicate whether specified player is settled on a location with specified harbor_type
        :param player: the player to check if he's settled on a location with given harbor-type
        :param harbor: harbor-type to check if given player is settled nearby
        :return: True if player settled near the harbor-type, false otherwise
        """
        for location in self._locations_by_harbors[harbor]:
            if self.is_colonised_by(player, location):
                return True
        return False

    def get_lands_to_place_robber_on(self) -> List[Land]:
        return [land for land in self._lands if land.identifier != self._robber_land.identifier]

    _vertices_rows = [
        [i for i in range(0, 3)],
        [i for i in range(3, 7)],
        [i for i in range(7, 11)],
        [i for i in range(11, 16)],
        [i for i in range(16, 21)],
        [i for i in range(21, 27)],
        [i for i in range(27, 33)],
        [i for i in range(33, 38)],
        [i for i in range(38, 43)],
        [i for i in range(43, 47)],
        [i for i in range(47, 51)],
        [i for i in range(51, 54)]
    ]
    _vertices = [v for vertices_row in _vertices_rows for v in vertices_row]

    @staticmethod
    def _compute_longest_road_length(g: networkx.Graph, u: Location, visited: Set[Path]):
        max_road_length = 0
        for v in g.neighbors(u):
            if (u, v) in visited or (v, u) in visited:
                continue
            visited.add((u, v))
            max_road_length = max(
                max_road_length,
                1 + Board._compute_longest_road_length(g, v, visited))
            visited.remove((u, v))
        return max_road_length

    def _create_and_shuffle_lands(self):
        land_numbers = [2, 12] + [i for i in range(3, 12) if i != 7] * 2
        land_resources = [Resource.Lumber, Resource.Wool, Resource.Grain
                          ] * 4 + [Resource.Brick, Resource.Ore] * 3
        self._shuffle(land_numbers)
        self._shuffle(land_resources)

        # get_lands_to_place_robber_on relies on the fact the 'desert' land.resource is None
        land_resources.append(None)
        land_numbers.append(0)

        ids = range(len(land_resources))
        locations = [[] for _ in range(len(land_resources))]
        surrounding_colonies = [[] for _ in range(len(land_resources))]

        lands = zip(land_resources, land_numbers, ids, locations, surrounding_colonies)
        self._lands = [Land(*land) for land in lands]

        self._robber_land = self._lands[-1]
        # Note how the robber location relies on the fact that the last
        # land in the list is the desert

    def _create_graph(self):
        self._roads_and_colonies = networkx.Graph()
        self._roads_and_colonies.add_nodes_from(Board._vertices)
        self._roads_and_colonies.add_edges_from(Board._create_edges())

    def _create_harbors(self):
        harbors = [Harbor.HarborBrick, Harbor.HarborLumber, Harbor.HarborWool, Harbor.HarborGrain, Harbor.HarborOre]
        self._shuffle(harbors)
        edges = self._get_harbors_edges()

        self._locations_by_harbors = {harbor: list(edge) for harbor, edge in zip(harbors, edges[0:len(harbors)])}
        self._locations_by_harbors[Harbor.HarborGeneric] = list(chain(*edges[len(harbors):]))

    def _get_harbors_edges(self):
        wrapping_edges = self._get_wrapping_edges()
        offsets = [4] * 3 + [3] * 6
        self._shuffle(offsets)
        indices = [offsets[0] - 2]
        for i in range(1, len(offsets)):
            indices.append(offsets[i] + indices[i - 1])
        return [wrapping_edges[i] for i in indices]

    def _get_wrapping_edges(self):
        u, v = (3, 0)
        wrapping_edges = [(u, v)]
        while (u, v) != (7, 3):
            assert len([w for w in self._roads_and_colonies.neighbors(v)
                        if w != u and self._is_wrapping_edge(v, w)]) == 1
            w = next(w for w in self._roads_and_colonies.neighbors(v) if w != u and self._is_wrapping_edge(v, w))
            wrapping_edges.append((v, w))
            u, v = v, w
        return wrapping_edges

    def _is_wrapping_edge(self, u, v):
        return len(self._roads_and_colonies[u][v][Board.lands]) == 1

    @staticmethod
    def _create_edges():
        edges = []
        for i in range(5):
            Board._create_row_edges(edges, i, i + 1, Board._vertices_rows, i % 2 == 0)
            Board._create_row_edges(edges, -i - 1, -i - 2, Board._vertices_rows, i % 2 == 0)
        Board._create_odd_rows_edges(edges, Board._vertices_rows[5], Board._vertices_rows[6])
        return edges

    @staticmethod
    def _create_row_edges(edges, i, j, vertices_rows, is_even_row):
        if is_even_row:
            Board._create_even_rows_edges(edges, vertices_rows[j], vertices_rows[i])
        else:
            Board._create_odd_rows_edges(edges, vertices_rows[j], vertices_rows[i])

    @staticmethod
    def _create_odd_rows_edges(edges, first_row, second_row):
        for edge in zip(second_row, first_row):
            edges.append(edge)

    @staticmethod
    def _create_even_rows_edges(edges, larger_row, smaller_row):
        for i in range(len(smaller_row)):
            edges.append((smaller_row[i], larger_row[i]))
            edges.append((smaller_row[i], larger_row[i + 1]))

    def _set_attributes(self):
        vertices_to_lands = self._create_vertices_to_lands_mapping()
        self._set_vertices_attributes(vertices_to_lands)
        self._set_edges_attributes(vertices_to_lands)
        self._set_lands_attributes(vertices_to_lands)

    def _create_vertices_to_lands_mapping(self):
        land_rows = [
            self._lands[0:3],
            self._lands[3:7],
            self._lands[7:12],
            self._lands[12:16],
            self._lands[16:19]
        ]
        vertices_rows_per_land_row = [
            Board._vertices_rows[0:3] + [Board._vertices_rows[3][1:-1]],
            Board._vertices_rows[2:5] + [Board._vertices_rows[5][1:-1]],
            Board._vertices_rows[4:8],
            [Board._vertices_rows[6][1:-1]] + Board._vertices_rows[7:10],
            [Board._vertices_rows[8][1:-1]] + Board._vertices_rows[9:12]
        ]
        vertices_map = {vertex: [] for vertex in Board._vertices}
        for vertices_rows, land_row in zip(vertices_rows_per_land_row, land_rows):
            Board._create_top_vertex_mapping(vertices_map, vertices_rows[0], land_row)
            Board._create_middle_vertex_mapping(vertices_map, vertices_rows[1], land_row)
            Board._create_middle_vertex_mapping(vertices_map, vertices_rows[2], land_row)
            Board._create_top_vertex_mapping(vertices_map, vertices_rows[3], land_row)
        return vertices_map

    def _set_vertices_attributes(self, vertices_to_lands):
        networkx.set_node_attributes(self._roads_and_colonies, vertices_to_lands, Board.lands)
        vertices_to_players = {v: (None, Colony.Uncolonised) for v in Board._vertices}
        networkx.set_node_attributes(self._roads_and_colonies, vertices_to_players, Board.player)

    def _set_edges_attributes(self, vertices_to_lands):
        for edge in self._roads_and_colonies.edges():
            lands_intersection = [land for land in vertices_to_lands[edge[0]]
                                  if land in vertices_to_lands[edge[1]]]
            edge_attributes = self._roads_and_colonies[edge[0]][edge[1]]
            edge_attributes[Board.lands] = lands_intersection
            edge_attributes[Board.player] = (None, Road.Unpaved)
            self._players_by_roads[path_key(edge)] = None

    @staticmethod
    def _set_lands_attributes(vertices_to_lands):
        for location, lands in vertices_to_lands.items():
            for land in lands:
                land.locations.append(location)

    @staticmethod
    def _create_top_vertex_mapping(vertices_map, vertices, lands):
        for vertex, land in zip(vertices, lands):
            vertices_map[vertex].append(land)

    @staticmethod
    def _create_middle_vertex_mapping(vertices_map, vertices, lands):
        vertices_map[vertices[0]].append(lands[0])
        vertices_map[vertices[-1]].append(lands[-1])

        for i in range(1, len(vertices[1:-1]) + 1):
            vertices_map[vertices[i]].append(lands[i - 1])
            vertices_map[vertices[i]].append(lands[i])
