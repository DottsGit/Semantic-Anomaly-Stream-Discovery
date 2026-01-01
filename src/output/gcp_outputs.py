"""Google Cloud Pub/Sub output for streaming events."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from src.analytics.flow_analyzer import OverallFlowStats
    from src.tracking.tracker import Track


class PubSubPublisher:
    """Publisher for streaming events to Google Cloud Pub/Sub."""

    def __init__(
        self,
        project_id: str,
        topic_name: str,
        batch_size: int = 100,
    ):
        """Initialize Pub/Sub publisher.

        Args:
            project_id: GCP project ID
            topic_name: Pub/Sub topic name
            batch_size: Number of messages to batch before publishing
        """
        self.project_id = project_id
        self.topic_name = topic_name
        self.batch_size = batch_size

        self._publisher = None
        self._topic_path: str = ""
        self._batch: list[dict] = []
        self._enabled = False

    def connect(self) -> bool:
        """Connect to Pub/Sub."""
        try:
            from google.cloud import pubsub_v1

            self._publisher = pubsub_v1.PublisherClient()
            self._topic_path = self._publisher.topic_path(self.project_id, self.topic_name)
            self._enabled = True
            logger.info(f"Connected to Pub/Sub topic: {self._topic_path}")
            return True

        except ImportError:
            logger.error("google-cloud-pubsub not installed. Run: pip install google-cloud-pubsub")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Pub/Sub: {e}")
            return False

    def publish_track_event(
        self,
        event_type: str,
        track: "Track",
        timestamp: float | None = None,
    ) -> None:
        """Publish a track event.

        Args:
            event_type: Event type (track_created, track_updated, track_lost)
            track: Track object
            timestamp: Optional timestamp (defaults to now)
        """
        if not self._enabled:
            return

        event = {
            "event_type": event_type,
            "timestamp": timestamp or datetime.utcnow().isoformat(),
            "track_id": track.track_id,
            "cluster_id": track.cluster_id,
            "cluster_label": track.cluster_label,
            "bbox": list(track.bbox),
            "center": list(track.center),
            "velocity": list(track.velocity),
            "speed": track.speed,
        }

        self._batch.append(event)

        if len(self._batch) >= self.batch_size:
            self._flush()

    def publish_stats(self, stats: "OverallFlowStats") -> None:
        """Publish flow statistics.

        Args:
            stats: Current flow statistics
        """
        if not self._enabled or self._publisher is None:
            return

        try:
            message = {
                "event_type": "flow_stats",
                "timestamp": datetime.utcnow().isoformat(),
                "data": stats.to_dict(),
            }

            data = json.dumps(message).encode("utf-8")
            future = self._publisher.publish(self._topic_path, data)
            future.result(timeout=5)  # Wait for publish

        except Exception as e:
            logger.error(f"Failed to publish stats: {e}")

    def _flush(self) -> None:
        """Flush batched messages."""
        if not self._batch or not self._enabled or self._publisher is None:
            return

        try:
            for event in self._batch:
                data = json.dumps(event).encode("utf-8")
                self._publisher.publish(self._topic_path, data)

            logger.debug(f"Published {len(self._batch)} events to Pub/Sub")
            self._batch = []

        except Exception as e:
            logger.error(f"Failed to flush Pub/Sub batch: {e}")

    def close(self) -> None:
        """Close the publisher."""
        self._flush()
        self._enabled = False
        logger.info("Closed Pub/Sub publisher")


class BigQueryWriter:
    """Writer for storing tracking data in BigQuery."""

    SCHEMA = [
        {"name": "timestamp", "type": "TIMESTAMP"},
        {"name": "track_id", "type": "INTEGER"},
        {"name": "cluster_id", "type": "INTEGER"},
        {"name": "cluster_label", "type": "STRING"},
        {"name": "bbox_x1", "type": "INTEGER"},
        {"name": "bbox_y1", "type": "INTEGER"},
        {"name": "bbox_x2", "type": "INTEGER"},
        {"name": "bbox_y2", "type": "INTEGER"},
        {"name": "center_x", "type": "INTEGER"},
        {"name": "center_y", "type": "INTEGER"},
        {"name": "velocity_x", "type": "FLOAT"},
        {"name": "velocity_y", "type": "FLOAT"},
        {"name": "speed", "type": "FLOAT"},
    ]

    def __init__(
        self,
        project_id: str,
        dataset: str,
        table: str,
        batch_size: int = 500,
    ):
        """Initialize BigQuery writer.

        Args:
            project_id: GCP project ID
            dataset: BigQuery dataset name
            table: BigQuery table name
            batch_size: Number of rows to batch before writing
        """
        self.project_id = project_id
        self.dataset = dataset
        self.table = table
        self.batch_size = batch_size

        self._client = None
        self._table_ref = None
        self._batch: list[dict] = []
        self._enabled = False

    def connect(self) -> bool:
        """Connect to BigQuery and ensure table exists."""
        try:
            from google.cloud import bigquery

            self._client = bigquery.Client(project=self.project_id)

            # Create dataset if not exists
            dataset_ref = self._client.dataset(self.dataset)
            try:
                self._client.get_dataset(dataset_ref)
            except Exception:
                dataset = bigquery.Dataset(dataset_ref)
                self._client.create_dataset(dataset)
                logger.info(f"Created BigQuery dataset: {self.dataset}")

            # Create table if not exists
            self._table_ref = dataset_ref.table(self.table)
            try:
                self._client.get_table(self._table_ref)
            except Exception:
                schema = [bigquery.SchemaField(f["name"], f["type"]) for f in self.SCHEMA]
                table = bigquery.Table(self._table_ref, schema=schema)
                self._client.create_table(table)
                logger.info(f"Created BigQuery table: {self.table}")

            self._enabled = True
            logger.info(f"Connected to BigQuery: {self.project_id}.{self.dataset}.{self.table}")
            return True

        except ImportError:
            logger.error("google-cloud-bigquery not installed. Run: pip install google-cloud-bigquery")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to BigQuery: {e}")
            return False

    def write_track(self, track: "Track", timestamp: float | None = None) -> None:
        """Write a track record.

        Args:
            track: Track object
            timestamp: Optional timestamp
        """
        if not self._enabled:
            return

        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "track_id": track.track_id,
            "cluster_id": track.cluster_id,
            "cluster_label": track.cluster_label,
            "bbox_x1": track.bbox[0],
            "bbox_y1": track.bbox[1],
            "bbox_x2": track.bbox[2],
            "bbox_y2": track.bbox[3],
            "center_x": track.center[0],
            "center_y": track.center[1],
            "velocity_x": track.velocity[0],
            "velocity_y": track.velocity[1],
            "speed": track.speed,
        }

        self._batch.append(row)

        if len(self._batch) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        """Flush batched rows to BigQuery."""
        if not self._batch or not self._enabled or self._client is None:
            return

        try:
            errors = self._client.insert_rows_json(self._table_ref, self._batch)
            if errors:
                logger.error(f"BigQuery insert errors: {errors}")
            else:
                logger.debug(f"Wrote {len(self._batch)} rows to BigQuery")

            self._batch = []

        except Exception as e:
            logger.error(f"Failed to flush BigQuery batch: {e}")

    def close(self) -> None:
        """Close the writer."""
        self._flush()
        self._enabled = False
        logger.info("Closed BigQuery writer")
