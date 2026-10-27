from pathlib import Path
import copy, gc, random, sys, unittest, weakref

ROOT = Path(__file__).resolve().parents[1]
import importlib.util
spec = importlib.util.spec_from_file_location('contest_fastcopy_test', ROOT/'core/envengine/fastcopy.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
deepcopy = module.deepcopy


class CoreTests(unittest.TestCase):
    def test_atomic_identity(self):
        for x in [None, True, 1, 2.5, 1+2j, 'abc', b'abc']:
            self.assertIs(deepcopy(x), x)

    def test_nested_independence(self):
        x = {1: {'pos': dict(x=1., y=2., z=3.)}, 2: [dict(v=1)]}
        y = deepcopy(x)
        self.assertEqual(x, y)
        y[1]['pos']['x'] = 9; y[2][0]['v'] = 9
        self.assertEqual(x[1]['pos']['x'], 1)
        self.assertEqual(x[2][0]['v'], 1)

    def test_cycle_and_alias(self):
        a = []; b = {'loop': a}; a.extend([b, b, a])
        c = deepcopy(a)
        self.assertIs(c[0], c[1]); self.assertIs(c[2], c)
        self.assertIs(c[0]['loop'], c)

    def test_tuple_cycle(self):
        a = []; t = (a,); a.append(t)
        result = deepcopy(t)
        self.assertIs(result[0][0], result)

    def test_shared_memo(self):
        x = {'v': [1]}; memo = {}
        y = deepcopy(x, memo)
        self.assertIs(deepcopy(x, memo), y)
        self.assertIs(deepcopy(x['v'], memo), y['v'])
        self.assertTrue(any(item is x for item in memo[id(memo)]))

    def test_custom_memo(self):
        class Memo(dict): pass
        x = [dict(a=1)]; memo = Memo()
        y = deepcopy(x, memo)
        self.assertEqual(y, x); self.assertIs(deepcopy(x, memo), y)

    def test_custom_copy_and_subclass(self):
        class Custom(list):
            def __deepcopy__(self, memo): return {'handled': True}
        self.assertEqual(deepcopy([Custom([1])]), [{'handled': True}])

    def test_callback_order(self):
        events = []
        class Marker:
            def __init__(self, name): self.name = name
            def __deepcopy__(self, memo): events.append(self.name); return self
        x = {Marker('key'): Marker('value')}
        deepcopy(x)
        self.assertEqual(events, ['value', 'key'])

    def test_dictionary_mutation_protocol(self):
        for copier in (copy.deepcopy, deepcopy):
            x = {}
            class Mutator:
                def __deepcopy__(self, memo): x['new'] = 1; return self
            x['first'] = Mutator()
            with self.assertRaises(RuntimeError): copier(x)

    def test_list_append_callback_protocol(self):
        for copier in (copy.deepcopy, deepcopy):
            x = []
            class Mutator:
                def __deepcopy__(self, memo): x.append(7); return 3
            x.append(Mutator())
            self.assertEqual(copier(x), [3, 7])

    def test_error_recovery(self):
        class Broken:
            def __deepcopy__(self, memo): raise ValueError('intentional')
        with self.assertRaises(ValueError): deepcopy([Broken()])
        self.assertEqual(deepcopy({'ok': [1]}), {'ok': [1]})

    def test_sdk_plain_dataclass(self):
        from dataclasses import make_dataclass
        Vector = make_dataclass("Vector", ["x", "y", "z"])
        v = Vector(1, 2, 3); x = [v, v]
        y = deepcopy(x)
        self.assertIs(y[0], y[1]); self.assertIsNot(y[0], v)
        self.assertEqual(vars(y[0]), vars(v))

    def test_no_leaked_sources_or_results(self):
        class Item: pass
        for _ in range(100):
            item = Item(); x = [item, {'data': item}]
            result = deepcopy(x); a = weakref.ref(item); b = weakref.ref(result[0])
            del item, x, result
            self.assertIsNone(a()); self.assertIsNone(b())

    def test_recursion_limit(self):
        root = []; tail = root
        for _ in range(2000): tail.append([]); tail = tail[0]
        with self.assertRaises(RecursionError): deepcopy(root)

    def test_randomized_trees(self):
        rng = random.Random(1307)
        def tree(depth):
            if depth == 0: return rng.uniform(-1e6, 1e6)
            return {str(i): [tree(depth-1), i] for i in range(rng.randrange(1, 5))}
        for _ in range(100):
            x = tree(3)
            self.assertEqual(deepcopy(x), copy.deepcopy(x))


    def test_prepopulated_memo_including_atoms(self):
        marker = object()
        self.assertIs(deepcopy(17, {id(17): marker}), marker)
        self.assertIs(deepcopy({'x': 17}, {id(17): marker})['x'], marker)

    def test_numpy_arrays(self):
        import numpy as np
        array = np.arange(6).reshape(2, 3)
        result = deepcopy([array, array])
        self.assertIs(result[0], result[1])
        self.assertFalse(np.shares_memory(array, result[0]))

    def test_no_stdlib_monkeypatch(self):
        self.assertEqual(copy.deepcopy.__module__, 'copy')

if __name__ == '__main__': unittest.main()
