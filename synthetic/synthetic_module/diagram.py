from numpy.random import uniform, choice
from random import randint, choice as rand_choice
import numpy as np
import re
import string
import copy
from .helper.seed import use_seed
from .helper.text import get_dictionary
from .element import AbstractElement
from PIL import Image, ImageDraw
from PIL import ImageFont
import math
POS_ELEMENT_OPACITY_RANGE = {
    "drawing": (220, 255),
    "glyph": (150, 255),
    "image": (150, 255),
    "table": (200, 255),
    "line": (120, 200),
    "table_word": (50, 200),
    "text": (200, 255),
    "diagram": (180, 255),
}

NEG_ELEMENT_OPACITY_RANGE = {
    "drawing": (0, 10),
    "glyph": (0, 10),
    "image": (0, 25),
    "table": (0, 25),
    "text": (0, 10),
    "diagram": (0, 25),
}
NEG_ELEMENT_BLUR_RADIUS_RANGE = (1, 2.5)
WIDTH_VALUES = [1, 2, 3, 4]

DIAGRAM_COLOR = (255, 100, 180)

COCENTRIC_CIRCLES_RATIO = 0.2
SAME_RADIUS_CIRCLES_RATIO = 0.1
COLORED_FREQ = 0.8
PROP_SINGLE_LETTER = 0.4

def computes_polygones(polygones, center_image, angle, pos, translation):
    rotation_matrix = np.array([[np.cos(angle * np.pi / 180), -np.sin(angle * np.pi / 180)],
                            [np.sin(angle * np.pi / 180), np.cos(angle * np.pi / 180)]])
    polygones_rotated = np.dot(polygones- center_image, rotation_matrix) + center_image
    polygones_rotated = polygones_rotated + translation
    polygones_rotated = polygones_rotated + pos
    return polygones_rotated

def compute_translation(canva_text_size,canva_text):
    xx = canva_text_size[0]
    yy = canva_text_size[1]
    nx,ny = canva_text.size
    tx = (nx - xx) /2.0
    ty = (ny - yy) /2.0
    return np.array([tx, ty])

def polygones_from_word(font,word):
    left, top, right, bottom = font.getbbox(word)            
    eps_w = (right - left) * 0.01          
    eps_h = (bottom - top) * 0.2
    x_pts = np.array([left-eps_w, right+eps_w, right+eps_w, left-eps_w])
    y_pts = np.array([bottom+eps_h, bottom+eps_h, top-eps_h, top-eps_h])
    return np.array([x_pts, y_pts]).T

def find_circle_center(p1, p2, p3):
    """Circle center from 3 points"""
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    if abs(det) < 1.0e-10:
        return (None, None)

    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    return np.array([cx, cy])


def get_angles_from_arc_points(p0, p_mid, p1):
    arc_center = find_circle_center(p0, p_mid, p1)
    arc_center = (arc_center[0], arc_center[1])
    start_angle = np.arctan2(p0[1] - arc_center[1], p0[0] - arc_center[0])
    end_angle = np.arctan2(p1[1] - arc_center[1], p1[0] - arc_center[0])
    mid_angle = np.arctan2(p_mid[1] - arc_center[1], p_mid[0] - arc_center[0])
    return start_angle, mid_angle, end_angle, arc_center


def find_circle_center_arr(p1, p2, p3):
    """Circle center from 3 points"""
    temp = p2[:, 0] ** 2 + p2[:, 1] ** 2
    bc = (p1[:, 0] ** 2 + p1[:, 1] ** 2 - temp) / 2
    cd = (temp - p3[:, 0] ** 2 - p3[:, 1] ** 2) / 2
    det = (p1[:, 0] - p2[:, 0]) * (p2[:, 1] - p3[:, 1]) - (p2[:, 0] - p3[:, 0]) * (
        p1[:, 1] - p2[:, 1]
    )

    # Handle the case where the determinant is close to zero
    mask = np.abs(det) < 1.0e-10
    det[mask] = 1.0  # Prevent division by zero
    bc[mask] = 0.0  # These arcs will have center at (0, 0)
    cd[mask] = 0.0

    cx = (bc * (p2[:, 1] - p3[:, 1]) - cd * (p1[:, 1] - p2[:, 1])) / det
    cy = ((p1[:, 0] - p2[:, 0]) * cd - (p2[:, 0] - p3[:, 0]) * bc) / det
    return np.stack([cx, cy], axis=-1)


def get_angles_from_arc_points_arr(p0, p_mid, p1):
    arc_center = find_circle_center_arr(p0, p_mid, p1)
    start_angle = np.arctan2(p0[:, 1] - arc_center[:, 1], p0[:, 0] - arc_center[:, 0])
    end_angle = np.arctan2(p1[:, 1] - arc_center[:, 1], p1[:, 0] - arc_center[:, 0])
    mid_angle = np.arctan2(
        p_mid[:, 1] - arc_center[:, 1], p_mid[:, 0] - arc_center[:, 0]
    )
    to_deg = lambda x: (x * 180 / np.pi) % 360
    start_angle = to_deg(start_angle)
    end_angle = to_deg(end_angle)
    mid_angle = to_deg(mid_angle)
    return start_angle, mid_angle, end_angle, arc_center


def gen_arc_from_p0_p1_radius(p0, p1, radius):
    p0, p1 = np.array(p0, dtype=np.float64), np.array(p1, dtype=np.float64)
    midpoint = (p0 + p1) / 2
    diff = p1 - p0
    dist = np.linalg.norm(diff)
    if dist != 0:
        unit_diff = diff / dist
        unit_perpendicular = np.array([-unit_diff[1], unit_diff[0]])
        offset = unit_perpendicular * np.sqrt(radius**2 - (dist / 2) ** 2)
    else:
        offset = np.array([0, radius])
    arc_center = midpoint + offset
    start_angle = np.arctan2(p0[1] - arc_center[1], p0[0] - arc_center[0])
    end_angle = np.arctan2(p1[1] - arc_center[1], p1[0] - arc_center[0])
    return start_angle, end_angle, arc_center


def is_valid_arc(
    arc_center,
    radius,
    start_angle,
    end_angle,
    width,
    height,
    min_angle=20,
    threshold_dist=5,
):
    angle1, angle2 = start_angle, end_angle
    if angle1 > angle2:
        angle2 += 2 * np.pi
    if np.abs(start_angle - end_angle) * 180 / np.pi % 360 < min_angle:
        return False

    testing_angles = np.linspace(angle1, angle2, 30)
    testing_pts = arc_center.reshape(2, 1) + radius * np.array(
        [np.cos(testing_angles), np.sin(testing_angles)]
    )
    if (
        (testing_pts[0, :] < 0).any()
        or (testing_pts[0, :] > width).any()
        or (testing_pts[1, :] < 0).any()
        or (testing_pts[1, :] > height).any()
    ):
        return False

    if np.linalg.norm(testing_pts[:, 0] - testing_pts[:, -1]) < threshold_dist:
        return False
    return True


class DiagramElement(AbstractElement):
    color = DIAGRAM_COLOR
    name = "diagram"

    @use_seed()
    def generate_content(self):
        dictionary, self.font_path,self.font_size = get_dictionary(self.parameters, self.height)
        self.diagram_position = self.parameters["diagram_position"]
        self.as_negative = self.parameters.get("as_negative", False)
        self.thickness_range = self.parameters.get("thickness_range", WIDTH_VALUES)
        self.blur_radius = (
            uniform(*NEG_ELEMENT_BLUR_RADIUS_RANGE) if self.as_negative else None
        )
       
        self.colored = choice([True, False], p=[COLORED_FREQ, 1 - COLORED_FREQ])
        self.colors = (
            tuple([randint(0, 60)] * 3)
            if not self.colored
            else tuple([randint(0, 255) for _ in range(3)])
        )
        self.text_colors = self.colors
        
        self.threshold_dist = self.parameters.get("threshold_dist", 20)
        self.number_circles = self.parameters.get(
            "number_circles", int(max((np.random.normal(randint(2, 10), 2)), 1))
        )
        self.number_arcs = self.parameters.get(
            "number_arcs", int(max((np.random.normal(randint(2, 20), 2)), 1))
        )
        self.number_lines = self.parameters.get(
            "number_lines", int(max((np.random.normal(randint(2, 20), 2)), 1))
        )
        self.number_words = self.parameters.get(
            "number_words", int(max((np.random.normal(randint(10, 40), 2)), 0))
        )

        self.flower_arcs = self.parameters.get(
            "flower_arcs", choice([True, False], p=[0.1, 0.9])
        )

        self.content_width = self.parameters.get("content_width", None)
        self.content_height = self.parameters.get("content_height", None)
        self.fill = choice([False, True], p=[0.8, 0.2])
        self.table, self.content_width, self.content_height = self._generate_diagram(
            dictionary, self.content_width, self.content_height
        )
        self.pos_x = randint(self.diagram_position[0], self.width - self.content_width)
        self.pos_y = randint(
            self.diagram_position[1], self.height - self.content_height
        )

    @use_seed()
    def _generate_diagram(self, dictionary, width=None, height=None):
        
        if width is None:
            width = randint(
                max(self.diagram_position[0], self.width // 3),
                self.width - 2 * self.diagram_position[0],
            )
        if height is None:
            height = randint(
                max(self.diagram_position[1], self.height // 3),
                self.height - 2 * self.diagram_position[1],
            )
        to_deg = lambda x: (x * 180 / np.pi) % 360
        circle_pos, circle_radius = [], []
        for i in range(self.number_circles):
            radius = np.random.uniform(
                min(10, min(width, height) // 2.1), min(width, height) // 2.1
            )
            center_x = np.random.uniform(radius, width - radius)
            center_y = np.random.uniform(radius, height - radius)
            circle_radius.append(radius)
            circle_pos.append((center_x, center_y))
            add_cocentric_circles = choice(
                [True, False], p=[COCENTRIC_CIRCLES_RATIO, 1 - COCENTRIC_CIRCLES_RATIO]
            )
            if add_cocentric_circles and radius > (min(width, height) // 6):
                num_circles_2 = 2 * randint(2, 8)

                for k in range(0, num_circles_2, 3):
                    new_radius = radius * np.random.uniform(
                        (k + 1) / num_circles_2, min((k + 2) / num_circles_2, 1)
                    )
                    circle_radius.append(new_radius)
                    circle_pos.append((center_x, center_y))

            add_same_radius_circles = choice(
                [True, False],
                p=[SAME_RADIUS_CIRCLES_RATIO, 1 - SAME_RADIUS_CIRCLES_RATIO],
            )
            if add_same_radius_circles:
                num_circles_2 = 2 * randint(2, 10)

                for k in range(0, num_circles_2, 2):
                    new_angle = np.random.uniform(
                        2 * np.pi * (k + 1) / num_circles_2,
                        2 * (k + 2) * np.pi / num_circles_2,
                    )
                    new_center_x = np.clip(
                        center_x + radius * np.cos(new_angle), radius, width - radius
                    )
                    new_center_y = np.clip(
                        center_y + radius * np.sin(new_angle), radius, height - radius
                    )

                    circle_radius.append(radius)
                    circle_pos.append((new_center_x, new_center_y))

        arc_centers, arc_radius, arc_angles = [], [], []
        to_deg = lambda x: (x * 180 / np.pi) % 360
        line_coords = []
        min_border = min(width, height)

        for i in range(self.number_arcs):
            shared_endpoints_arcs = choice([True, False], p=[0.2, 0.8])
            shared_point_arcs = choice([True, False], p=[0.2, 0.8])
            horizontal_arc = choice([True, False], p=[0.2, 0.8])
            vertical_arc = choice(
                [True, False], p=[0.25, 0.75]
            )  # FIXME fix the distribution
            p0 = np.array(
                [np.random.uniform(10, width - 10), np.random.uniform(10, height - 10)]
            )
            if horizontal_arc:
                p1 = np.array([np.random.randint(10, width - 10), p0[1]])
            elif vertical_arc:
                p1 = np.array([p0[0], np.random.uniform(10, height - 10)])

            p1 = np.array(
                [np.random.uniform(10, width - 10), np.random.uniform(10, height - 10)]
            )
            try:
                radius = np.random.uniform(
                    max(
                        min_border // 20, np.linalg.norm((p1 - p0)) // 1.9 + 10
                    ),  # cant have a radius smaller than the mid distance between the two points
                    min_border * 2,
                )
            except ValueError as e:
                print(e)
                radius = np.linalg.norm((p1 - p0)) // 1.5

            start_angle, end_angle, arc_center = gen_arc_from_p0_p1_radius(
                p0, p1, radius
            )
            valid_arc = is_valid_arc(
                arc_center, radius, start_angle, end_angle, width, height
            )
            if not valid_arc:
                continue

            start_angle, end_angle = to_deg(start_angle), to_deg(end_angle)

            arc_radius.append(radius)
            arc_centers.append(arc_center)
            arc_angles.append((start_angle, end_angle))
            if shared_endpoints_arcs:
                num_arcs = randint(2, 8)
                new_radius = radius
                for k in range(num_arcs):
                    if uniform() < 0.5:
                        new_p0, new_p1 = p1, p0
                    else:
                        new_p0, new_p1 = p0, p1

                    new_radius = new_radius * np.random.uniform(1.5, 3)

                    (
                        new_start_angle,
                        new_end_angle,
                        new_arc_center,
                    ) = gen_arc_from_p0_p1_radius(new_p0, new_p1, new_radius)
                    valid_arc = is_valid_arc(
                        new_arc_center,
                        new_radius,
                        new_start_angle,
                        new_end_angle,
                        width,
                        height,
                    )
                    if not valid_arc:
                        break

                    new_start_angle, new_end_angle = to_deg(new_start_angle), to_deg(
                        new_end_angle
                    )
                    arc_centers.append(new_arc_center)
                    arc_radius.append(new_radius)
                    arc_angles.append((new_start_angle, new_end_angle))
            if shared_point_arcs:
                num_arcs = randint(2, 8)
                for k in range(num_arcs):
                    interpolation_factor = np.random.uniform(0.2, 0.8)
                    new_radius = radius * np.random.uniform(1.0, 1.4)
                    new_start_angle = (
                        interpolation_factor * start_angle * np.pi / 180
                        + (1 - interpolation_factor) * end_angle * np.pi / 180
                    )

                    new_p0 = arc_center + radius * np.array(
                        [np.cos(new_start_angle), np.sin(new_start_angle)]
                    )  # choosing point that lies on the arc
                    new_p1 = np.array(
                        [np.random.randint(0, width), np.random.randint(0, height)]
                    )
                    if uniform() < 0.5:
                        new_p0, new_p1 = new_p1, new_p0
                    new_radius = max(
                        radius, np.linalg.norm(new_p0 - new_p1) / 2 + 10
                    ) * np.random.uniform(1.0, 1.6)
                    (
                        new_start_angle,
                        new_end_angle,
                        new_arc_center,
                    ) = gen_arc_from_p0_p1_radius(new_p0, new_p1, new_radius)

                    valid_arc = is_valid_arc(
                        new_arc_center,
                        new_radius,
                        new_start_angle,
                        new_end_angle,
                        width,
                        height,
                    )
                    if not valid_arc:
                        continue
                    new_start_angle, new_end_angle = to_deg(new_start_angle), to_deg(
                        new_end_angle
                    )
                    arc_centers.append(new_arc_center)
                    arc_radius.append(new_radius)
                    arc_angles.append((new_start_angle, new_end_angle))

        if self.flower_arcs:
            num_arcs = randint(2, 10)
            radius = randint(
                min(40, min(width, height) // 2.1), min(width, height) // 2.1
            )
            center_x = np.random.randint(radius, width - radius)
            center_y = np.random.randint(radius, height - radius)
            circle_center = np.array([center_x, center_y]).reshape(1, 2)
            if choice([True, False]):  # show circle
                circle_radius.append(radius)
                circle_pos.append((center_x, center_y))
            start_angles = np.linspace(0, np.pi, num_arcs) + np.random.normal(
                0, 2, num_arcs
            )

            end_angles = (np.pi + start_angles) + np.random.normal(0, 2, num_arcs)

            start_pts = (
                radius * np.array([np.cos(start_angles), np.sin(start_angles)]).T
                + circle_center
            )
            end_pts = (
                radius * np.array([np.cos(end_angles), np.sin(end_angles)]).T
                + circle_center
            )
            noise_angle = np.random.uniform(0, 2 * np.pi)
            noise_distance = np.random.uniform(radius // 8, radius // 2)
            noise_distance = 0
            mid_point = circle_center + noise_distance * np.array(
                [np.cos(noise_angle), np.sin(noise_angle)]
            )
            mid_pts = np.repeat(mid_point.reshape(1, 2), num_arcs, axis=0)
            current_arc_centers = find_circle_center_arr(start_pts, mid_pts, end_pts)
            radii = np.linalg.norm(current_arc_centers - start_pts, axis=1)

            for k in range(num_arcs):
                p0, p1 = start_pts[k, :], end_pts[k, :]
                radius = radii[k]
                start_angle, end_angle, arc_center = gen_arc_from_p0_p1_radius(
                    p0, p1, radius
                )

                def is_large_arc(rad_angle):
                    if rad_angle[0] <= np.pi:
                        return not (
                            rad_angle[0] < rad_angle[1] < (np.pi + rad_angle[0])
                        )
                    return (rad_angle[0] - np.pi) < rad_angle[1] < rad_angle[0]

                if is_large_arc((start_angle, end_angle)):
                    start_angle, end_angle = end_angle, start_angle
                valid_arc = is_valid_arc(
                    arc_center, radius, start_angle, end_angle, width, height
                )
                if not valid_arc:
                    continue
                start_angle, end_angle = to_deg(start_angle), to_deg(end_angle)
                arc_centers.append(arc_center)
                arc_radius.append(radius)
                arc_angles.append((start_angle, end_angle))

        arc_centers = np.array(arc_centers)
        arc_angles = np.array(arc_angles)
        arc_radius = np.array(arc_radius)
        circle_pos = np.array(circle_pos)
        circle_radius = np.array(circle_radius)

        if len(circle_pos) > self.number_circles:
            keep_indices = np.random.choice(
                len(circle_pos), size=self.number_circles, replace=False
            )
            circle_pos = circle_pos[keep_indices]
            circle_radius = circle_radius[keep_indices]

        if len(arc_centers) > self.number_arcs:
            keep_indices = np.random.choice(
                len(arc_centers), size=self.number_arcs, replace=False
            )
            arc_centers = arc_centers[keep_indices]
            arc_radius = arc_radius[keep_indices]
            arc_angles = arc_angles[keep_indices]

        line_coords = []
        for i in range(self.number_lines):
            length = randint(10, min(width // 2 - 1, height // 2 - 1))
            coords_x = np.random.randint(length, width - length)
            coords_y = np.random.randint(length, height - length)

            angle = np.random.uniform(0, 2 * np.pi)
            x_length = length * np.abs(np.cos(angle))
            y_length = length * np.sin(angle)
            direction = choice([1, -1])
            coords = (
                coords_x - x_length,
                coords_y - direction * y_length,
                coords_x + x_length,
                coords_y + direction * y_length,
            )
            line_coords.append(coords)
        shared_point_lines = choice([True, False], p=[0.1, 0.9])
        if shared_point_lines:
            length = randint(10, min(width // 2 - 1, height // 2 - 1))
            start_x = np.random.randint(length, width - length)
            start_y = np.random.randint(length, height - length)

            for num_line in range(randint(2, 10)):
                if choice([True, False], p=[0.2, 0.8]):
                    length = np.random.uniform(0.8, 0.9) * length
                angle = np.random.uniform(0, 2 * np.pi)
                x_length = length * np.abs(np.cos(angle))
                y_length = length * np.sin(angle)
                direction = choice([1, -1])
                coords = (
                    start_x,
                    start_y,
                    start_x + x_length,
                    start_y + direction * y_length,
                )
                line_coords.append(coords)
        horizontal_lines = choice([True, False], p=[0.1, 0.9])
        vertical_lines = choice([True, False], p=[0.1, 0.9])
        if horizontal_lines:
            for num_line in range(randint(1, 10)):
                start_x = np.random.randint(0, width)
                start_y = np.random.randint(0, height)
                end_x = np.random.randint(0, width)
                end_y = start_y
                line_coords.append((start_x, start_y, end_x, end_y))
        if vertical_lines:
            for num_line in range(randint(1, 10)):
                start_x = np.random.randint(0, width)
                start_y = np.random.randint(0, height)
                end_x = start_x
                end_y = np.random.randint(0, height)
                line_coords.append((start_x, start_y, end_x, end_y))
        line_coords = np.array(line_coords)

        if len(line_coords) > self.number_lines:
            keep_indices = np.random.choice(
                len(line_coords), size=self.number_lines, replace=False
            )
            line_coords = line_coords[keep_indices]

        list_objects = [] 
        if len(circle_pos) > 0:
            list_objects.append('circle')
        if len(arc_centers) > 0:
            list_objects.append('arc')
        if len(line_coords) > 0:
            list_objects.append('line')
        


        words, word_positions = [], []
        word_objects = []
        word_idx_objects = []
        list_circles = np.arange(len(circle_pos))
        list_arcs = np.arange(len(arc_centers))
        list_lines = np.arange(len(line_coords))


        circle_pos_bis = copy.deepcopy(circle_pos)
        circle_radius_bis = copy.deepcopy(circle_radius)

        arc_centers_bis = copy.deepcopy(arc_centers)
        arc_radius_bis = copy.deepcopy(arc_radius)
        arc_angles_bis = copy.deepcopy(arc_angles)

        line_coords_bis = copy.deepcopy(line_coords)
        word_radius = []
        word_angle = []
        word_center = []
        word_line_coord = []
      
        is_latin = self.parameters.get("is_latin", False)
        if is_latin:
            letters_dict = [ch for ch in string.ascii_letters + '123456789' if ch != 'O']
            words_dict = []
            for sentence in dictionary:
                all_words = re.findall(r'\b[a-zA-Z0-9]+\b', sentence)
                filtered_words = [word for word in all_words if len(word) > 1]
                words_dict.extend(filtered_words)
            if not words_dict:
                words_dict = [w for w in dictionary if len(w) > 1] or list(letters_dict)
        for i in range(self.number_words):
            if is_latin:
                word_as_letter = choice([True, False], p=[0.6, 0.4])
                if word_as_letter:
                    word = choice(letters_dict)
                else:
                    word_as_long = choice([True, False], p=[0.5, 0.5])
                    if word_as_long:
                        n_letter = randint(2, 10)
                        word = ' '.join(rand_choice(words_dict) for _ in range(n_letter))
                    else:
                        word = rand_choice(words_dict)
                    uppercase = choice([True, False])
                    if uppercase:
                        word = word.upper()
            else:
                word_as_number = choice([True, False], p=[0.1, 0.9])
                if word_as_number:
                    n_letter = randint(1, 4)
                    word = f"{randint(0, 10**n_letter - 1):,}"
                else:
                    word = rand_choice(dictionary)
                    uppercase = choice([True, False])
                    if uppercase:
                        word = word.upper()
            if len(word) > 0:
                list_objects = [] 
                if len(list_circles) > 0:
                    list_objects.append('circle')
                if len(list_arcs) > 0:
                    list_objects.append('arc')
                if len(list_lines) > 0:
                    list_objects.append('line')
                if len(list_objects) == 0:
                    continue
    
                attached_to_object = choice(list_objects)
                if attached_to_object == 'circle':

                    circle_idx = np.random.randint(0, len(list_circles))
                    idx_to_remove = circle_idx
                    circle_idx = list_circles[circle_idx]
    


                    center = circle_pos_bis[circle_idx]
                    radius = circle_radius_bis[circle_idx]
                    ## pos somewhere on the circle
                    angle = np.random.uniform(0, 2 * np.pi)
                    offset = choice([np.random.randint(radius-10,radius-5),np.random.randint(radius+5,radius+10)])
                    x = center[0] + offset * np.cos(angle)
                    y = center[1] + offset * np.sin(angle)
                    word_positions.append((x, y))

                    word_radius.append(radius)
                    word_angle.append([0,2*np.pi])
                    word_center.append(center)
                    word_line_coord.append([0,0,0,0])
                    
                    list_circles = np.delete(list_circles, idx_to_remove)
                    word_idx_objects.append(circle_idx)


                elif attached_to_object == 'arc':
                    
                    arc_idx = np.random.randint(0, len(list_arcs))
                    idx_to_remove = arc_idx
                    arc_idx = list_arcs[arc_idx]

                    center = arc_centers_bis[arc_idx]
                    radius = arc_radius_bis[arc_idx]
                    start_angle_, end_angle_ = arc_angles_bis[arc_idx]
                    angle = np.random.uniform(start_angle_, end_angle_)
                    offset = choice([np.random.randint(radius-10,radius-5),np.random.randint(radius+5,radius+10)])
                    x = center[0] + offset * np.cos(np.radians(angle))
                    y = center[1] + offset * np.sin(np.radians(angle))
                    word_radius.append(radius)
                    word_angle.append([start_angle_, end_angle_])
                    word_center.append(center)

                    word_positions.append((x, y))
                    word_line_coord.append([0,0,0,0])

                    list_arcs = np.delete(list_arcs, idx_to_remove)
                    word_idx_objects.append(arc_idx)

                elif attached_to_object == 'line':
                    line_idx = np.random.randint(0, len(list_lines))
                    idx_to_remove = line_idx
                    line_idx = list_lines[line_idx]
                    x1, y1, x2, y2 = line_coords_bis[line_idx]
                    x = np.random.uniform(x1, x2)
     
                    y = min(y1,y2) + choice([1,-1]) * np.random.uniform(2,5)
                    word_positions.append((x, y))

       
                    list_lines = np.delete(list_lines, idx_to_remove)
                    word_radius.append(0)
                    word_angle.append([0,2*np.pi])
                    word_center.append([0,0])
                    word_line_coord.append([x1,y1,x2,y2])
                    word_idx_objects.append(line_idx)
                words.append(word)
                word_objects.append(attached_to_object)
        word_radius = np.array(word_radius)
        word_angle = np.array(word_angle)
        word_center = np.array(word_center)
        word_line_coord = np.array(word_line_coord)
        word_idx_objects = np.array(word_idx_objects)
        return (
            {
                "circle_pos": circle_pos,
                "circle_radius": circle_radius,
                "line_coords": line_coords,
                "arc_centers": arc_centers,
                "arc_radius": arc_radius,
                "arc_angles": arc_angles,
                "words": words,
                "word_positions": word_positions,
                "word_objects": word_objects,
                "word_radius": word_radius,
                "word_angle": word_angle,
                "word_center": word_center,
                "word_line_coord": word_line_coord,
                "word_idx_objects": word_idx_objects
            },
            width,
            height,
        )

    @use_seed()
    def to_image(self):
        canvas = Image.new("RGBA", self.size)
        draw = ImageDraw.Draw(canvas)
        if self.fill:
            opacity = randint(40, 80)
            fill_color = tuple(randint(0, 255) for _ in range(3)) + (opacity,)
        else:
            fill_color = None
        prev_circle_radius = 0
        prev_circle_pos = np.array([0, 0])
        
        new_circle_pos = []
        new_circle_radius = []

        for circle_pos, circle_radius in zip(
            self.table["circle_pos"], self.table["circle_radius"]
        ):

            colors = (
                tuple([randint(0, 60)] * 3)
                if not self.colored
                else tuple([randint(0, 255) for _ in range(3)])
            )
            # opacity = 255
            keep_same_params = (prev_circle_radius == circle_radius) or (
                prev_circle_pos == circle_pos
            ).all()
            if not keep_same_params:
                params = {
                    "fill": fill_color,
                    "outline": colors,
                    "width": rand_choice(self.thickness_range),
                }

            center = [self.pos_x + circle_pos[0], self.pos_y + circle_pos[1]]
            new_circle_pos.append(center)
            new_circle_radius.append(circle_radius)
           
            shape = [
                center[0] - circle_radius,
                center[1] - circle_radius,
                center[0] + circle_radius,
                center[1] + circle_radius,
            ]
            fill_color = None  # only fill the first circle to not overlap
            draw.ellipse(shape, **params)
            prev_circle_radius = circle_radius
            prev_circle_pos = circle_pos

        new_arc_centers = []
        new_arc_radius = []
        new_arc_angles = []

        for arc_center, arc_radius, arc_angles in zip(
            self.table["arc_centers"],
            self.table["arc_radius"],
            self.table["arc_angles"],
        ):
            colors = (
                tuple([randint(0, 60)] * 3)
                if not self.colored
                else tuple([randint(0, 255) for _ in range(3)])
            )
            params = {
                "fill": colors,
                "width": rand_choice(self.thickness_range),
            }

            center = [self.pos_x + arc_center[0], self.pos_y + arc_center[1]]
            shape = [
                center[0] - arc_radius,
                center[1] - arc_radius,
                center[0] + arc_radius,
                center[1] + arc_radius,
            ]

            draw.arc(shape, start=arc_angles[0], end=arc_angles[1], **params)
            new_arc_centers.append(center)
            new_arc_radius.append(arc_radius)
            new_arc_angles.append(arc_angles)

        new_line_coords = []
        for line_coords in self.table["line_coords"]:
            colors = (
                tuple([randint(0, 60)] * 3)
                if not self.colored
                else tuple([randint(0, 255) for _ in range(3)])
            )
            params = {
                "fill": colors,
                "width": rand_choice(self.thickness_range),
            }

            draw.line(
                [
                    self.pos_x + line_coords[0],
                    self.pos_y + line_coords[1],
                    self.pos_x + line_coords[2],
                    self.pos_y + line_coords[3],
                ],
                **params,
            )
            new_line_coords.append([self.pos_x + line_coords[0], self.pos_y + line_coords[1],self.pos_x + line_coords[2], self.pos_y + line_coords[3]])
            assert self.pos_x + line_coords[0] < self.width, f"{line_coords}"
            assert self.pos_y + line_coords[1] < self.height, f"{line_coords}"
            assert self.pos_x + line_coords[2] < self.width, f"{line_coords}"
            assert self.pos_y + line_coords[3] < self.height, f"{line_coords}"
        list_polygones = []
        list_words = []
        for word, pos,word_object,word_radius,word_angle,word_center,word_line_coord,word_idx_objects in zip(self.table["words"], self.table["word_positions"],self.table["word_objects"],self.table["word_radius"],self.table["word_angle"],self.table["word_center"],self.table["word_line_coord"],self.table["word_idx_objects"]):
            opacity = randint(*POS_ELEMENT_OPACITY_RANGE['text'])
            stroke_width = np.random.randint(0, 4)
            # sample number betzeen 0 qnd 1
            if np.random.uniform(0, 1) < PROP_SINGLE_LETTER:
                word = word[0]
                font_size = self.font_size * np.random.randint(1,3)
            else:
                font_size =  self.font_size 
            colors_alpha = self.text_colors + (opacity,)
            colors = (
                tuple([randint(0, 60)] * 3)
                if not self.colored
                else tuple([randint(0, 255) for _ in range(3)])
            )
            pos = pos[0] + self.pos_x, pos[1] + self.pos_y
            if len(word)>14:
                word = word[:14]
            

            if word_object == 'line':

                font = ImageFont.truetype(self.font_path, size=font_size)
                canva_text_size = font.getsize(word)
                canva_text = Image.new('L',canva_text_size, color = 0)
                draw_text = ImageDraw.Draw(canva_text)

                if ('chinese' in self.font_path or 'arabic' in self.font_path or 'hebrew' in self.font_path):
                    draw_text.text((0, 0), word, font=font,  fill=255, direction = 'rtl')
                else:
                    draw_text.text((0, 0), word, font=font,  fill=255)
                line_coord__ = new_line_coords[word_idx_objects]
                angle = np.arctan2(line_coord__[3] - line_coord__[1], line_coord__[2] - line_coord__[0])
                angle = angle * 180 / np.pi * rand_choice([-1,1])


                polygones = polygones_from_word(font,word)
                center_image = np.array([canva_text_size[0]//2, canva_text_size[1]//2])
                
                canva_text = canva_text.rotate(angle, expand=True)
                translation = compute_translation(canva_text_size,canva_text)
                polygones= computes_polygones(polygones, center_image, angle, pos, translation)
                if (polygones[:,0].min() < 0) or (polygones[:,0].max() > self.size[0]) or (polygones[:,1].min() < 0) or (polygones[:,1].max() > self.size[1]):
                            continue
                if ('chinese' in self.font_path or 'arabic' in self.font_path or 'hebrew' in self.font_path)  and not any(char.isdigit() for char in word):
                    

                    pp = polygones
                    x_coords = pp[:,0]
                    y_coords = pp[:,1]

                    bottom_x = x_coords[:len(x_coords)//2]
                    bottom_y = y_coords[:len(y_coords)//2]

                    top_x = x_coords[len(x_coords)//2:]
                    top_y = y_coords[len(y_coords)//2:]
                    
                    bottom_x =  bottom_x[::-1]
                    bottom_y = bottom_y[::-1]

                    top_x = top_x[::-1]
                    top_y = top_y[::-1]

                    x_coords = np.concatenate([bottom_x,top_x])
                    y_coords = np.concatenate([bottom_y,top_y])
                    polygones = np.array([x_coords,y_coords]).T
                list_polygones.append(polygones)
                list_words.append(word)
                canvas.paste(colors_alpha, (int(pos[0]), int(pos[1])), mask = canva_text)


            elif word_object == 'arc' and word_angle[1] - word_angle[0]  < np.pi*0.5:
                
                font = ImageFont.truetype(self.font_path, size=font_size)
                
                #draw text with correct opacity opacity
                
                polygones = polygones_from_word(font,word)    
                polygones = polygones + pos
                if (polygones[:,0].min() < 0) or (polygones[:,0].max() > self.size[0]) or (polygones[:,1].min() < 0) or (polygones[:,1].max() > self.size[1]):
                            continue
                if ('chinese' in self.font_path or 'arabic' in self.font_path or 'hebrew' in self.font_path)  and not any(char.isdigit() for char in word):
                    pp = polygones
                    x_coords = pp[:,0]
                    y_coords = pp[:,1]

                    bottom_x = x_coords[:len(x_coords)//2]
                    bottom_y = y_coords[:len(y_coords)//2]

                    top_x = x_coords[len(x_coords)//2:]
                    top_y = y_coords[len(y_coords)//2:]
                    
                    bottom_x =  bottom_x[::-1]
                    bottom_y = bottom_y[::-1]

                    top_x = top_x[::-1]
                    top_y = top_y[::-1]

                    x_coords = np.concatenate([bottom_x,top_x])
                    y_coords = np.concatenate([bottom_y,top_y])
                    polygones = np.array([x_coords,y_coords]).T
                list_polygones.append(polygones)
                list_words.append(word)
                draw.text(pos, word, font=font, fill = colors_alpha, stroke_width = stroke_width)
                if ('chinese' in self.font_path or 'arabic' in self.font_path or 'hebrew' in self.font_path):
                    draw.text(pos, word, font=font, fill = colors_alpha, stroke_width = stroke_width, direction = 'rtl')
                else:
                    draw.text(pos, word, font=font, fill = colors_alpha, stroke_width = stroke_width)
                

            elif word_radius < 50:


                font = ImageFont.truetype(self.font_path, size=font_size)
                
                polygones = polygones_from_word(font,word)
                polygones = polygones + pos
                if (polygones[:,0].min() < 0) or (polygones[:,0].max() > self.size[0]) or (polygones[:,1].min() < 0) or (polygones[:,1].max() > self.size[1]):
                    continue
                if ('chinese' in self.font_path or 'arabic' in self.font_path or 'hebrew' in self.font_path)  and not any(char.isdigit() for char in word):
                        pp = polygones
                        x_coords = pp[:,0]
                        y_coords = pp[:,1]

                        bottom_x = x_coords[:len(x_coords)//2]
                        bottom_y = y_coords[:len(y_coords)//2]

                        top_x = x_coords[len(x_coords)//2:]
                        top_y = y_coords[len(y_coords)//2:]
                        
                        bottom_x =  bottom_x[::-1]
                        bottom_y = bottom_y[::-1]

                        top_x = top_x[::-1]
                        top_y = top_y[::-1]

                        x_coords = np.concatenate([bottom_x,top_x])
                        y_coords = np.concatenate([bottom_y,top_y])
                        polygones = np.array([x_coords,y_coords]).T
                list_polygones.append(polygones)
                list_words.append(word)
                draw.text(pos, word, font=font, fill=colors_alpha, stroke_width = stroke_width)
                if ('chinese' in self.font_path or 'arabic' in self.font_path or 'hebrew' in self.font_path):
                    draw.text(pos, word, font=font, fill=colors_alpha, stroke_width = stroke_width, direction = 'rtl')
                else:
                    draw.text(pos, word, font=font, fill=colors_alpha, stroke_width = stroke_width)
                
            else:
                if word_object == 'circle':

                    radius__ = new_circle_radius[word_idx_objects]
                    center__ = new_circle_pos[word_idx_objects]
                else:
                    radius__ = new_arc_radius[word_idx_objects]
                    center__ = new_arc_centers[word_idx_objects]
                    angles__ = new_arc_angles[word_idx_objects] 
                if len(word) > 4:
                    word = word[:4]
                radius = radius__ 


                if ('chinese' in self.font_path or 'arabic' in self.font_path or 'hebrew' in self.font_path):
                    continue
                else:
                    font = ImageFont.truetype(self.font_path, size=font_size)

                if word_object == 'circle':
                    start_angle = np.random.uniform(0,360)
                else:
                    
                    start_angle = angles__[0]
      
   
                width, heigth= font.getsize(word)
                width = int(1.2 * width)
                radius = radius + heigth+5


                end_angle = (360 * width) / (2 * np.pi * radius) + start_angle
              
                center = center__ 

        

                angle_delta = (end_angle - start_angle) / len(word)
                reversed = False 
                upside_down = 1 
                polygones_word = []
                if not reversed:
                    for i in range(len(word)):
                        angle = start_angle + i * angle_delta

                        x = center[0] + radius * math.cos(math.radians(angle))
                        y = center[1] + radius * math.sin(math.radians(angle))
                        ## write in black

                        char_image_size = font.getsize(word[i])
                        char_image = Image.new('L', char_image_size, color = 0)
                        char_draw = ImageDraw.Draw(char_image)
                        char_draw.text((0, 0), word[i], font=font, fill=255, stroke_width = stroke_width)
                        angle_char =90*3 - angle

                        polygones = polygones_from_word(font,word[i])
                        center_image = np.array([char_image_size[0]//2, char_image_size[1]//2])
                        char_image = char_image.rotate(upside_down * angle_char , expand=True)
                        translation = compute_translation(char_image_size,char_image)
                        polygones = computes_polygones(polygones, center_image,upside_down * angle_char, np.array([int(x), int(y)]), translation)
                        # check if polygones is outside the image
                        if (polygones[:,0].min() < 0) or (polygones[:,0].max() > self.size[0]) or (polygones[:,1].min() < 0) or (polygones[:,1].max() > self.size[1]):
                            continue
                        else:
                            canvas.paste(colors_alpha, (int(x), int(y)), mask =char_image)
                            polygones_word.append(polygones) ##REMOVE OUTSIDE
                    if len(polygones_word) == 0:
                        continue
                    polygones_word = np.array(polygones_word) #shape (len(word), 4, 2)

                    x_pts_bottom = polygones_word[:,:2,0].flatten()
                    x_pts_bottom = x_pts_bottom[::2]
                    x_pts_bottom = np.concatenate([x_pts_bottom, [polygones_word[-1,1,0 ]] ])  
                                
                    y_pts_bottom = polygones_word[:,:2,1].flatten()
                    y_pts_bottom = y_pts_bottom[::2]
                    y_pts_bottom = np.concatenate([y_pts_bottom, [polygones_word[-1,1,1 ]] ])


                    x_pts_top_a = polygones_word[:,2,0].flatten()
                    x_pts_top_a = x_pts_top_a[::-1]
                    x_pts_top = np.concatenate([x_pts_top_a, [polygones_word[0,-1,0]]])

                    y_pts_top_a = polygones_word[:,2,1].flatten()
                    y_pts_top_a = y_pts_top_a[::-1]
                    y_pts_top = np.concatenate([y_pts_top_a, [polygones_word[0,-1,1]]])


                    x_pts = np.concatenate([x_pts_bottom, x_pts_top])
                    y_pts = np.concatenate([y_pts_bottom, y_pts_top])
                    polygones = np.array([x_pts,y_pts]).T

                    list_polygones.append(polygones)
                    list_words.append(word)
 

                else:
                    for i in range(len(word), 0, -1):
                        angle = start_angle + i * angle_delta

                        x = center[0] + radius * math.cos(math.radians(angle))
                        y = center[1] + radius * math.sin(math.radians(angle))
                        ## write in black
                        char_image = Image.new('L', (font.getsize(word[i-1])), color = 0)
                        char_draw = ImageDraw.Draw(char_image)
                        char_draw.text((0, 0), word[i-1], font=font, fill=255, stroke_width = stroke_width)
   
                        angle_char =90*3 - angle

                        char_image = char_image.rotate(upside_down*angle_char , expand=True)
                        canvas.paste(colors_alpha, (int(x), int(y)), mask =char_image)

        self.list_polygones = list_polygones
        self.list_words = list_words
        return canvas

    def get_annotation(self):
        centers, circle_radii, lines, arcs = [], [], [], []
        x_min = float('inf')
        x_max = 0. 
        y_min = float('inf')
        y_max = 0.
        for pp in self.list_polygones:

            min_x_line = min(pp[:,0])
            max_x_line = max(pp[:,0])
            min_y_line = min(pp[:,1])
            max_y_line = max(pp[:,1])


            if min_x_line < x_min:
                x_min = min_x_line
            if max_x_line > x_max:
                x_max = max_x_line
            if min_y_line < y_min:
                y_min = min_y_line
            if max_y_line > y_max:
                y_max = max_y_line

        return {'polygones': self.list_polygones,'words':self.list_words, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min,'y_max': y_max}
