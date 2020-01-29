"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime

import numpy
from parameterized import parameterized
from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.constants import LOG_SCHEMA
from sponge_bob_magic.splitters.log_splitter import (
    ColdUsersExtractingSplitter, LogSplitByDateSplitter,
    LogSplitRandomlySplitter)
from sponge_bob_magic.utils import get_distinct_values_in_column


class TestLogSplitByDateSplitter(PySparkTest):
    def test_split(self):
        log = self.spark.createDataFrame(
            data=[
                ["user1", "item1", datetime(2019, 9, 12), "day", 1.0],
                ["user1", "item2", datetime(2019, 9, 13), "night", 2.0],
                ["user2", "item1", datetime(2019, 9, 14), "day", 3.0],
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0],
                ["user3", "item1", datetime(2019, 9, 16), "day", 5.0],
                ["user1", "item3", datetime(2019, 9, 17), "night", 1.0]
            ],
            schema=LOG_SCHEMA
        )
        train, predict_input, test = (
            LogSplitByDateSplitter(
                False, False,
                datetime(2019, 9, 15))
            .split(log)
        )

        true_train = self.spark.createDataFrame(
            data=[
                ["user1", "item1", datetime(2019, 9, 12), "day", 1.0],
                ["user1", "item2", datetime(2019, 9, 13), "night", 2.0],
                ["user2", "item1", datetime(2019, 9, 14), "day", 3.0]
            ],
            schema=LOG_SCHEMA
        )
        true_test = self.spark.createDataFrame(
            data=[
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0],
                ["user3", "item1", datetime(2019, 9, 16), "day", 5.0],
                ["user1", "item3", datetime(2019, 9, 17), "night", 1.0]
            ],
            schema=LOG_SCHEMA
        )
        self.assertSparkDataFrameEqual(true_train, train)
        self.assertSparkDataFrameEqual(train, predict_input)
        self.assertSparkDataFrameEqual(true_test, test)

        train, predict_input, test = (
            LogSplitByDateSplitter(
                True, False,
                datetime(2019, 9, 15))
            .split(log)
        )
        true_test = self.spark.createDataFrame(
            data=[
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0],
                ["user3", "item1", datetime(2019, 9, 16), "day", 5.0]
            ],
            schema=LOG_SCHEMA
        )
        self.assertSparkDataFrameEqual(true_train, train)
        self.assertSparkDataFrameEqual(train, predict_input)
        self.assertSparkDataFrameEqual(true_test, test)

        train, predict_input, test = (
            LogSplitByDateSplitter(
                False, True,
                datetime(2019, 9, 15))
            .split(log)
        )
        true_test = self.spark.createDataFrame(
            data=[
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0],
                ["user1", "item3", datetime(2019, 9, 17), "night", 1.0]
            ],
            schema=LOG_SCHEMA
        )
        self.assertSparkDataFrameEqual(true_train, train)
        self.assertSparkDataFrameEqual(train, predict_input)
        self.assertSparkDataFrameEqual(true_test, test)

        train, predict_input, test = (
            LogSplitByDateSplitter(
                True, True,
                datetime(2019, 9, 15))
            .split(log)
        )
        true_test = self.spark.createDataFrame(
            data=[
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0]
            ],
            schema=LOG_SCHEMA
        )
        self.assertSparkDataFrameEqual(true_train, train)
        self.assertSparkDataFrameEqual(train, predict_input)
        self.assertSparkDataFrameEqual(true_test, test)


class TestLogSplitRandomlySplitter(PySparkTest):
    @parameterized.expand([
        # test_size, drop_cold_items, drop_cold_users
        (0.0, False, False),
        (0.3, False, False),
        (0.8, False, False),
        (1.0, False, False),
        (0.5, True, False),
        (0.6, True, False),
        (0.7, True, False),
        (0.22, False, True),
        (0.35, False, True),
        (0.65, False, True),
        (0.42, True, True),
        (0.75, True, True),
        (0.95, True, True),
    ])
    def test_split(self, test_size, drop_cold_items, drop_cold_users):
        seed = 1234
        log = self.spark.createDataFrame(
            data=[
                ["user1", "item4", datetime(2019, 9, 12), "day", 1.0],
                ["user4", "item1", datetime(2019, 9, 12), "day", 1.0],
                ["user4", "item2", datetime(2019, 9, 13), "night", 2.0],
                ["user2", "item4", datetime(2019, 9, 14), "day", 3.0],
                ["user2", "item1", datetime(2019, 9, 14), "day", 3.0],
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0],
                ["user3", "item1", datetime(2019, 9, 16), "day", 5.0],
                ["user3", "item4", datetime(2019, 9, 16), "day", 5.0],
                ["user1", "item3", datetime(2019, 9, 17), "night", 1.0]
            ],
            schema=LOG_SCHEMA
        )

        train, test_input, test = (
            LogSplitRandomlySplitter(
                drop_cold_items=drop_cold_items,
                drop_cold_users=drop_cold_users,
                test_size=test_size,
                seed=seed)
            .split(log)
        )

        if not drop_cold_items and not drop_cold_users:
            self.assertSparkDataFrameEqual(log, train.union(test))
            self.assertSparkDataFrameEqual(
                log, test.union(test_input)
            )
            self.assertEqual(
                test.count(), numpy.ceil(log.count() * test_size)
            )

        if drop_cold_items:
            test_items = get_distinct_values_in_column(test, "item_id")
            test_input_items = get_distinct_values_in_column(test, "item_id")
            train_items = get_distinct_values_in_column(test, "item_id")

            self.assertSetEqual(test_items, test_input_items)
            self.assertSetEqual(test_items, train_items)
        if drop_cold_users:
            test_users = get_distinct_values_in_column(test, "user_id")
            test_input_users = get_distinct_values_in_column(test, "user_id")
            train_users = get_distinct_values_in_column(test, "user_id")

            self.assertSetEqual(test_users, test_input_users)
            self.assertSetEqual(test_users, train_users)


class TestColdUsersExtractingSplitter(PySparkTest):
    def test_split(self):
        log = self.spark.createDataFrame(
            data=[
                ["user2", "item4", datetime(2019, 9, 14), "day", 3.0],
                ["user2", "item1", datetime(2019, 9, 14), "day", 3.0],
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0],
                ["user1", "item4", datetime(2019, 9, 12), "day", 1.0],
                ["user4", "item1", datetime(2019, 9, 12), "day", 1.0],
                ["user4", "item2", datetime(2019, 9, 13), "night", 2.0],
                ["user3", "item1", datetime(2019, 9, 16), "day", 5.0],
                ["user3", "item4", datetime(2019, 9, 16), "day", 5.0],
                ["user1", "item3", datetime(2019, 9, 17), "night", 1.0]
            ],
            schema=LOG_SCHEMA
        )

        train, test_input, test = (
            ColdUsersExtractingSplitter(
                False, False,
                test_size=1 / 4)
            .split(log=log)
        )

        self.assertSparkDataFrameEqual(
            self.spark.createDataFrame(data=[], schema=LOG_SCHEMA),
            test_input)

        true_train = self.spark.createDataFrame(
            data=[
                ["user1", "item4", datetime(2019, 9, 12), "day", 1.0],
                ["user4", "item1", datetime(2019, 9, 12), "day", 1.0],
                ["user4", "item2", datetime(2019, 9, 13), "night", 2.0],
                ["user2", "item4", datetime(2019, 9, 14), "day", 3.0],
                ["user2", "item1", datetime(2019, 9, 14), "day", 3.0],
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0],
            ],
            schema=LOG_SCHEMA
        )
        true_test = self.spark.createDataFrame(
            data=[
                ["user3", "item1", datetime(2019, 9, 16), "day", 5.0],
                ["user3", "item4", datetime(2019, 9, 16), "day", 5.0],
            ],
            schema=LOG_SCHEMA
        )

        self.assertSparkDataFrameEqual(train, true_train)
        self.assertSparkDataFrameEqual(test, true_test)
