import copy
import json
import unittest
from communication_copy import CommunicationDict, _Snapshot


def sample():
    return {1: {'pos_ecf': {'x': 1., 'y': 2., 'z': 3.},
                'velEcf': {'x': 4., 'y': 5., 'z': 6.}},
            2: {'pos_ecf': {'x': 7., 'y': 8., 'z': 9.}}}


class CommunicationCopyTests(unittest.TestCase):
    def test_nested_independent_writes_after_deepcopy(self):
        source = CommunicationDict.from_plain(sample())
        first = copy.deepcopy(source); second = copy.deepcopy(source)
        self.assertIs(dict.__getitem__(first, 1), dict.__getitem__(second, 1))
        first[1]['pos_ecf']['x'] = 100
        self.assertEqual(second[1]['pos_ecf']['x'], 1.)
        self.assertEqual(source[1]['pos_ecf']['x'], 1.)
        third = copy.deepcopy(first)
        first[1]['pos_ecf']['x'] = 200
        self.assertEqual(third[1]['pos_ecf']['x'], 100)

    def test_snapshot_detaches_from_original(self):
        plain = sample(); snapshot = CommunicationDict.from_plain(plain)
        plain[1]['pos_ecf']['x'] = 100
        self.assertEqual(snapshot[1]['pos_ecf']['x'], 1.)

    def test_accessors_never_expose_shared_mutable_state(self):
        accessors = [lambda d:d.get(1), lambda d:d.setdefault(1),
                     lambda d:next(iter(d.values())), lambda d:next(iter(d.items()))[1],
                     lambda d:dict(d)[1], lambda d:{**d}[1], lambda d:dict(d.items())[1],
                     lambda d:d.copy()[1], lambda d:copy.copy(d)[1],
                     lambda d:(d | {})[1], lambda d:({} | d)[1], lambda d:d.pop(1)]
        for accessor in accessors:
            with self.subTest(accessor=accessor):
                source=CommunicationDict.from_plain(sample()); owner=copy.deepcopy(source)
                value=accessor(owner); value['pos_ecf']['x']=99
                self.assertEqual(source[1]['pos_ecf']['x'],1.)

    def test_shallow_copy_keeps_ordinary_aliases(self):
        owner=CommunicationDict.from_plain(sample()); shallow=owner.copy()
        shallow[1]['pos_ecf']['x']=99
        self.assertEqual(owner[1]['pos_ecf']['x'],99)

    def test_views_are_live_and_match_dict(self):
        owner=CommunicationDict.from_plain(sample()); keys=owner.keys(); values=owner.values()
        owner[3]={'new':True}
        self.assertEqual(len(values),3);self.assertIn(3,keys)
        self.assertEqual(dict(owner.items()),dict(owner))
        self.assertEqual(owner,sample() | {3:{'new':True}})

    def test_json_roundtrip(self):
        owner=CommunicationDict.from_plain(sample())
        for sort in (True,False):
            self.assertEqual(json.dumps(owner,sort_keys=sort),json.dumps(sample(),sort_keys=sort))

    def test_cycles_aliases_and_user_mutations(self):
        owner=CommunicationDict.from_plain(sample()); extra=[]
        owner['a']=extra;owner['b']=extra;owner['self']=owner
        cloned=copy.deepcopy(owner)
        self.assertIs(cloned['a'],cloned['b']);self.assertIs(cloned['self'],cloned)
        self.assertIsNot(cloned['a'],extra)
        owner.update({4:{'fresh':True}});owner |= {5:{'fresh':False}}
        self.assertEqual(owner.popitem(),(5,{'fresh':False}))
        self.assertEqual(owner.pop('absent',3),3)
        with self.assertRaises(KeyError):owner.pop('absent')
        owner.clear();self.assertEqual(owner,{})

    def test_private_tree_rejects_writes_and_unsupported_values(self):
        tree=_Snapshot.freeze(sample())
        with self.assertRaises(TypeError):tree[1]=0
        with self.assertRaises(TypeError):tree[1]['pos_ecf']['x']=0
        with self.assertRaises(TypeError):CommunicationDict.from_plain({'unexpected':[]})


if __name__=='__main__': unittest.main()
