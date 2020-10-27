# -*- coding:utf-8 -*-
"""Test function for vega.core.trainer.trigger."""
import unittest

from vega.core.trainer.trigger import Trigger


@Trigger.activate('train')
def train():
    """Activate log trigger."""
    print("train")


@Trigger.activate('valid')
def valid():
    """Activate valid trigger."""
    print("valid")


@Trigger.register(['train', 'valid'])
class LoggerTrigger(Trigger):
    """Register logger trigger."""

    def before(self):
        """Execute before function."""
        print("Before train")

    def after(self):
        """Execute after function"""
        print("After train")


@Trigger.register('train')
class ProfileTrigger(Trigger):
    """Register Profile trigger."""

    def before(self):
        """Execute before function."""
        print("Profile: Before train")

    def after(self):
        """Execute after function"""
        print("Profile: After train")


class TestTrigger(unittest.TestCase):
    """Test trigger wrapper."""

    def test_trigger(self):
        """Test trigger"""
        train()
        valid()
        print(Trigger.__triggers__)


if __name__ == "__main__":
    unittest.main()
