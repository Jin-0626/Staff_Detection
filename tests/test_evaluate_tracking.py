import unittest
from evaluate_tracking import evaluate, match_boxes


def obj(identity='staff1', box=None):
    return dict(identity=identity, bbox_xyxy=box or [0,0,10,10])


def track(tid, box=None):
    box = box or [0,0,10,10]
    return dict(track_id=tid, bbox_xyxy=box, centre_xy=[(box[0]+box[2])/2,(box[1]+box[3])/2])


class EvaluationTests(unittest.TestCase):
    def test_new_id_every_frame_counts_switches(self):
        labels = {'frames': [dict(frame=i, objects=[obj()]) for i in range(1,5)]}
        rows = [dict(frame=i, detections=[obj()], tracks=[track(i)]) for i in range(1,5)]
        report = evaluate(rows, labels)
        self.assertEqual(report['identity_switches'], 3)
        self.assertEqual(report['frame_recall'], 1)

    def test_misses_and_gap(self):
        rows = [dict(frame=1,detections=[],tracks=[]),
                dict(frame=2,detections=[obj()],tracks=[]),
                dict(frame=3,detections=[obj()],tracks=[track(1)])]
        report = evaluate(rows, {'frames':[dict(frame=i,objects=[obj()]) for i in range(1,4)]})
        self.assertEqual(report['missed_gt_detections'], 1)
        self.assertEqual(report['detected_gt_without_matching_track'], 1)
        self.assertEqual(report['missed_presence_intervals'], [[1,2]])
        self.assertAlmostEqual(report['frame_recall'], 1/3)

    def test_wrong_person_presence_not_object_match(self):
        rows = [dict(frame=1,detections=[obj(box=[20,20,30,30])],tracks=[track(1,[20,20,30,30])])]
        report = evaluate(rows, {'frames':[dict(frame=1,objects=[obj()])]})
        self.assertEqual(report['frame_recall'], 1)
        self.assertEqual(report['object_recall'], 0)

    def test_unreviewed_frames_not_negative_or_switches(self):
        rows = [dict(frame=i,detections=[obj()],tracks=[track(i)]) for i in range(1,4)]
        report = evaluate(rows, {'frames':[dict(frame=1,objects=[obj()]),dict(frame=3,objects=[obj()])]})
        self.assertEqual(report['reviewed_frames'], 2)
        self.assertEqual(report['identity_switches'], 0)

    def test_switch_after_missed_frame(self):
        rows = [dict(frame=1,detections=[obj()],tracks=[track(1)]),
                dict(frame=2,detections=[],tracks=[]),
                dict(frame=3,detections=[obj()],tracks=[track(2)])]
        report = evaluate(rows, {'frames':[dict(frame=i,objects=[obj()]) for i in range(1,4)]})
        self.assertEqual(report['identity_switches'], 1)

    def test_absence_and_missing_prediction(self):
        report = evaluate([dict(frame=1,detections=[],tracks=[])], {'frames':[dict(frame=1,objects=[])]})
        self.assertEqual(report['frame_tn'], 1)
        self.assertIsNone(report['frame_recall'])
        with self.assertRaises(ValueError):
            evaluate([], {'frames':[dict(frame=1,objects=[])]})

    def test_one_to_one(self):
        self.assertEqual(len(match_boxes([obj(),obj('staff2')],[track(1)],.5)), 1)


if __name__ == '__main__':
    unittest.main()
