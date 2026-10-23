import unittest
from collections import namedtuple

from form_logic import (
    LEFT_SIDE, MSG_BACK, MSG_DEEPER, MSG_GOOD_FORM, MSG_GOOD_REP, MSG_NOT_COUNTED, MSG_READY, RIGHT_SIDE,
    PushupCounter, calculate_angle, side_points,
)

Landmark = namedtuple("Landmark", "x y visibility")

STRAIGHT_HIPS = 175


def do_rep(counter, bottom=80, hip=STRAIGHT_HIPS):
    """Feed a top -> bottom -> top sequence of elbow angles."""
    for elbow in (170, 130, bottom, 130, 170):
        counter.update(elbow, hip)


class CalculateAngleTests(unittest.TestCase):
    def test_right_angle(self):
        self.assertAlmostEqual(calculate_angle((0, 1), (0, 0), (1, 0)), 90)

    def test_straight_line(self):
        self.assertAlmostEqual(calculate_angle((-1, 0), (0, 0), (1, 0)), 180)

    def test_always_reflex_folded(self):
        self.assertAlmostEqual(calculate_angle((1, 0), (0, 0), (0, -1)), 90)
        self.assertAlmostEqual(calculate_angle((1, 1), (0, 0), (1, -1)), 90)


class SidePointsTests(unittest.TestCase):
    def make_landmarks(self, left_vis, right_vis):
        lms = [Landmark(0.5, 0.5, 0.0)] * 33
        for i in LEFT_SIDE:
            lms[i] = Landmark(0.1, 0.2, left_vis)
        for i in RIGHT_SIDE:
            lms[i] = Landmark(0.9, 0.8, right_vis)
        return lms

    def test_picks_more_visible_side_and_scales_to_pixels(self):
        points = side_points(self.make_landmarks(0.3, 0.9), 640, 480)
        self.assertAlmostEqual(points["elbow"][0], 0.9 * 640)
        self.assertAlmostEqual(points["elbow"][1], 0.8 * 480)
        self.assertEqual(set(points), {"shoulder", "elbow", "wrist", "hip", "knee"})

    def test_none_when_neither_side_visible(self):
        self.assertIsNone(side_points(self.make_landmarks(0.2, 0.4), 640, 480))

    def test_pixel_scaling_fixes_aspect_distortion(self):
        # A true 90-degree elbow on a 640x480 frame, with both forearm and upper arm at 45 degrees.
        lms = [Landmark(0, 0, 0.0)] * 33
        shoulder, elbow, wrist, hip, knee = LEFT_SIDE
        lms[shoulder] = Landmark(200 / 640, 100 / 480, 1)
        lms[elbow] = Landmark(300 / 640, 200 / 480, 1)
        lms[wrist] = Landmark(400 / 640, 100 / 480, 1)
        lms[hip] = Landmark(200 / 640, 100 / 480, 1)
        lms[knee] = Landmark(100 / 640, 100 / 480, 1)
        p = side_points(lms, 640, 480)
        self.assertAlmostEqual(calculate_angle(p["shoulder"], p["elbow"], p["wrist"]), 90)
        # Using the raw normalized values (what the old code did) gives the wrong answer.
        raw = calculate_angle((lms[shoulder].x, lms[shoulder].y), (lms[elbow].x, lms[elbow].y),
                              (lms[wrist].x, lms[wrist].y))
        self.assertGreater(abs(raw - 90), 5)


class PushupCounterTests(unittest.TestCase):
    def setUp(self):
        self.counter = PushupCounter()

    def test_starts_ready(self):
        self.assertEqual((self.counter.reps, self.counter.stage, self.counter.feedback), (0, None, MSG_READY))

    def test_full_rep_counts_on_the_way_up(self):
        for elbow in (170, 130, 80):
            self.counter.update(elbow, STRAIGHT_HIPS)
        self.assertEqual((self.counter.reps, self.counter.stage), (0, "down"))
        self.counter.update(170, STRAIGHT_HIPS)
        self.assertEqual(self.counter.reps, 1)
        self.assertEqual(self.counter.feedback, MSG_GOOD_REP)

    def test_multiple_reps(self):
        for _ in range(3):
            do_rep(self.counter)
        self.assertEqual(self.counter.reps, 3)

    def test_no_rep_before_starting_at_top(self):
        for elbow in (80, 170):
            self.counter.update(elbow, STRAIGHT_HIPS)
        self.assertEqual(self.counter.reps, 0)

    def test_partial_rep_says_go_deeper(self):
        do_rep(self.counter, bottom=110)
        self.assertEqual(self.counter.reps, 0)
        self.assertEqual(self.counter.feedback, MSG_DEEPER)

    def test_sagging_hips_during_rep_not_counted(self):
        for elbow, hip in ((170, 175), (130, 175), (80, 140), (130, 175), (170, 175)):
            self.counter.update(elbow, hip)
        self.assertEqual(self.counter.reps, 0)
        self.assertEqual(self.counter.feedback, MSG_NOT_COUNTED)

    def test_bad_hips_only_at_top_does_not_spoil_next_rep(self):
        self.counter.update(170, 140)
        self.assertEqual(self.counter.feedback, MSG_BACK)
        self.counter.update(170, STRAIGHT_HIPS)
        self.assertEqual(self.counter.feedback, MSG_GOOD_FORM)
        do_rep(self.counter)
        self.assertEqual(self.counter.reps, 1)

    def test_rep_after_bad_rep_counts(self):
        do_rep(self.counter, hip=140)
        do_rep(self.counter)
        self.assertEqual(self.counter.reps, 1)

    def test_update_from_points(self):
        top = {"shoulder": (0, 0), "elbow": (0, 10), "wrist": (0, 20), "hip": (30, 0), "knee": (60, 0)}
        self.counter.update_from_points(top)
        self.assertEqual(self.counter.stage, "up")


if __name__ == "__main__":
    unittest.main()
