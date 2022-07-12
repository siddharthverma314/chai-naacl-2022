from neural_chat.logger import simpleloggable


def test_loggable():
    @simpleloggable
    class Test:
        def __init__(self, nolog, _log):
            pass

        def test_log(self):
            self.log("a", 1)
            self.log("b", 2)

    a = Test("bye", "hi")
    a.test_log()
    assert a.log_hyperparams() == {"log": "hi"}
    assert a.log_epoch() == {"a": 1, "b": 2}


def test_nested_loggable():
    @simpleloggable
    class Test1:
        def __init__(self, nolog1, _log1):
            pass

        def test_log(self):
            self.log("a1", 1)
            self.log("b1", 2)

    @simpleloggable
    class Test2(Test1):
        def __init__(self, nolog2, _log2, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def test_log(self):
            super().test_log()
            self.log("a2", 3)
            self.log("b2", 4)

    @simpleloggable
    class Test3(Test2):
        def __init__(self, nolog3, _log3, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def test_log(self):
            super().test_log()
            self.log("a3", 5)
            self.log("b3", 6)

    t = Test3("bye3", "hi3", "bye2", "hi2", nolog1="bye1", _log1="hi1")
    t.test_log()
    assert t.log_hyperparams() == {"log1": "hi1", "log2": "hi2", "log3": "hi3"}
    assert t.log_epoch() == {"a1": 1, "b1": 2, "a2": 3, "b2": 4, "a3": 5, "b3": 6}
