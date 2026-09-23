# -*- coding: utf-8 -*-
"""The one place that knows how URFD is split in half.

URFD is the only accuracy surface this project has never tuned against, which makes it the
only one that can settle a question. That is worth protecting: a decision made by reading all
hundred clips spends its independence, and the project has paid for that mistake once already.
So every configuration is scored clip by clip, half A is what a choice may be made on, and
half B is only ever read afterwards to confirm the choice.

**The split takes sequences two at a time, not odd and even.** URFD alternates its fall types
by sequence number -- odd-numbered sequences are falls from standing, even-numbered ones are
falls out of a chair -- so an odd/even split does not compare two samples of one task, it
compares two different tasks, and a configuration can win on one and lose on the other for
reasons that have nothing to do with whether it is better. Taking them in pairs puts both
types in both halves. Both cameras of one incident carry the same sequence number, so an
incident never straddles the split either.

This lived in three copies with two different implementations before it lived here. They
agreed, but nothing made them agree.
"""


def clip_index(name):
    """`fall-07-cam0-rgb.mp4` -> 7, the URFD sequence number the split keys on."""
    digits = ''.join(c if c.isdigit() else ' ' for c in name).split()
    return int(digits[0]) if digits else 0


def half(name):
    """-> 'A' (may be chosen on) or 'B' (confirms only) for a URFD clip filename."""
    return 'A' if ((clip_index(name) - 1) // 2) % 2 == 0 else 'B'


def in_half_a(name):
    return half(name) == 'A'


def in_half_b(name):
    return half(name) == 'B'
