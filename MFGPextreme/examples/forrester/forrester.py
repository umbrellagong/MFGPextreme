from emukit.test_functions import forrester


def f_l(x):
    if x.ndim == 2:
        return forrester.forrester_low(x).reshape(-1)
    elif x.ndim == 1:
        return forrester.forrester_low(x.reshape(-1,1)).item()


def f_h(x):
    if x.ndim == 2:
        return forrester.forrester(x).reshape(-1)
    elif x.ndim == 1:
        return forrester.forrester(x).item()