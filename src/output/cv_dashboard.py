"""Tabbed OpenCV dashboard for per-class cluster visualization and signal monitoring.

This module provides a secondary OpenCV window with two tabs:
1. Cluster View - Per-class 2D scatter plot with dropdown selection
2. Signal Monitor - Live time-series graph with class checkboxes
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from datetime import datetime

from src.clustering.class_aware_clusterer import ClassAwareClusterer, get_class_color


@dataclass
class SignalHistory:
    """Tracks detection counts over time for a class."""
    
    class_name: str
    is_anomaly: bool
    counts: deque = field(default_factory=lambda: deque(maxlen=120))  # 2 minutes at 1Hz
    max_count: int = 0

    
    @property
    def label(self) -> str:
        """Display label for this signal."""
        if self.is_anomaly:
            return f"{self.class_name} (anomaly)"
        return self.class_name


class CVDashboard:
    """Tabbed OpenCV dashboard with cluster visualization and signal monitor."""
    
    TAB_CLUSTERS = 0
    TAB_SIGNALS = 1
    TAB_ANOMALIES = 2
    
    # UI Constants
    TAB_HEIGHT = 35
    DROPDOWN_HEIGHT = 30
    CHECKBOX_SIZE = 16
    PADDING = 10
    
    def __init__(self, width: int = 600, height: int = 500, window_name: str = "SASD - Dashboard"):
        """Initialize dashboard.
        
        Args:
            width: Window width
            height: Window height  
            window_name: OpenCV window name
        """
        self.width = width
        self.height = height
        self.window_name = window_name
        
        # Tab state
        self.current_tab = self.TAB_CLUSTERS
        
        # Tab 1: Cluster view state
        self.available_classes: list[str] = []
        self.selected_class_idx = 0
        self.dropdown_open = False
        
        # Tab 2: Signal monitor state  
        self.signal_checkboxes: dict[str, bool] = {}  # label -> is_checked
        self.signal_history: dict[str, SignalHistory] = {}  # label -> history
        
        # Colors
        self.bg_color = (30, 30, 30)
        self.tab_color = (50, 50, 50)
        self.tab_active_color = (80, 80, 80)
        self.text_color = (220, 220, 220)
        self.dropdown_bg = (45, 45, 45)
        
        # Mouse callback setup
        self._mouse_x = 0
        self._mouse_y = 0
        self._mouse_clicked = False
        self._mouse_down = False
        self._mouse_wheel_delta = 0
        self._window_initialized = False
        
        # Scroll state
        self.signal_scroll_y = 0
        self.signal_scroll_max = 0
        self.ROW_HEIGHT = 24
        
        # Tab 3: Anomaly Review state
        self.anomalies: deque = deque(maxlen=50) # (dict with id, class, image, time)
        self.selected_anomaly_idx = -1
        self.anomaly_scroll_y = 0
        self.anomaly_scroll_max = 0
        self.anomaly_filter_class: str | None = None
        self.anomaly_dropdown_open: bool = False
        self.is_dragging_scrollbar: bool = False
        self.drag_start_y: int = 0
        self.drag_start_scroll_y: int = 0


        
    def _init_window(self) -> None:
        """Initialize OpenCV window with mouse callback."""
        if self._window_initialized:
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self._window_initialized = True
        
    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse events."""
        self._mouse_x = x
        self._mouse_y = y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_clicked = True
            self._mouse_down = True
        elif event == cv2.EVENT_LBUTTONUP:
            self._mouse_down = False
            self.is_dragging_scrollbar = False # Stop dragging globally
        elif event == cv2.EVENT_MOUSEWHEEL:
            # flags > 0 means scroll up, < 0 means scroll down
            # Standard convention: wheel up (positive) -> scroll up (decrease offset)
            # wheel down (negative) -> scroll down (increase offset)
            velocity = 1 if cv2.getMouseWheelDelta(flags) > 0 else -1
            self._mouse_wheel_delta = velocity
            
    def update(
        self, 
        class_aware_clusterer: ClassAwareClusterer | None,
        active_counts: dict[str, int],
    ) -> None:
        """Update dashboard with latest data.
        
        Args:
            class_aware_clusterer: The per-class clusterer with embeddings
            active_counts: Dict of class_label -> active count (e.g., "car": 5, "car (anomaly)": 1)
        """
        # Update available classes from clusterer
        if class_aware_clusterer and class_aware_clusterer._fitted:
            self.available_classes = sorted(class_aware_clusterer._clusterers.keys())
            
        # Update signal history
        for label, count in active_counts.items():
            # Determine base class and anomaly status
            is_anomaly = "(anomaly)" in label
            base_class = label.replace(" (anomaly)", "")
            
            # Ensure both variants exist in history
            normal_label = base_class
            anomaly_label = f"{base_class} (anomaly)"
            
            for lbl, is_anom in [(normal_label, False), (anomaly_label, True)]:
                if lbl not in self.signal_history:
                    self.signal_history[lbl] = SignalHistory(
                        class_name=base_class,
                        is_anomaly=is_anom,
                    )
                    # Default: disable all signals by default (user request)
                    self.signal_checkboxes[lbl] = False
            
            # Update counts (active_counts has current values)
            # We must be careful: active_counts might only have one of them
            # actual updating happens below in the loop over signal_history
            
        # Update all histories
        for label, history in self.signal_history.items():
            count = active_counts.get(label, 0)
            history.counts.append(count)
            if count > history.max_count:
                history.max_count = count
                
    def register_anomaly(
        self, 
        track_id: int, 
        class_name: str, 
        image: np.ndarray
    ) -> None:
        """Register a new anomaly event."""
        # Add to deque (left = newest)
        self.anomalies.appendleft({
            "id": track_id,
            "class": class_name,
            "image": image.copy(),
            "time": datetime.now()
        })
        # If nothing selected, select the new one (index 0)
        if self.selected_anomaly_idx == -1:
            self.selected_anomaly_idx = 0
        # If something selected, shift index by 1 so we stay on same item? 
        # No, auto-selecting new one is better for "live" feel, 
        # OR keep selection on specific item? 
        # Standard behavior: If I'm looking at item X, don't jump to New Item Y.
        # But if we shift items down, index 0 becomes index 1.
        elif self.selected_anomaly_idx >= 0:
            self.selected_anomaly_idx += 1
            if self.selected_anomaly_idx >= len(self.anomalies):
                self.selected_anomaly_idx = len(self.anomalies) - 1
                
    def draw(self, class_aware_clusterer: ClassAwareClusterer | None) -> np.ndarray:
        """Draw the dashboard.
        
        Args:
            class_aware_clusterer: Clusterer for accessing per-class data
            
        Returns:
            OpenCV image (BGR)
        """
        # Create canvas
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = self.bg_color
        
        # Handle mouse interactions
        self._handle_ui_interactions()
        
        # Draw tab bar
        self._draw_tabs(canvas)
        
        # Draw tab content
        content_y = self.TAB_HEIGHT + 5
        content_area = canvas[content_y:, :]
        
        if self.current_tab == self.TAB_CLUSTERS:
            self._draw_cluster_tab(content_area, class_aware_clusterer)
        elif self.current_tab == self.TAB_SIGNALS:
            self._draw_signal_tab(content_area)
        else:
            self._draw_anomaly_tab(content_area)
            
        # Reset click state
        self._mouse_clicked = False
        
        return canvas
        
    def _handle_ui_interactions(self) -> None:
        """Process mouse clicks and scroll on UI elements."""
        # Check if we have any interaction (click, scroll, or dragging)
        if not self._mouse_clicked and self._mouse_wheel_delta == 0 and not self.is_dragging_scrollbar:
            return
            
        # Tab clicks (only on actual click)
        if self._mouse_clicked and self._mouse_y < self.TAB_HEIGHT:
            tab_width = self.width // 3
            if self._mouse_x < tab_width:
                self.current_tab = self.TAB_CLUSTERS
            elif self._mouse_x < 2 * tab_width:
                self.current_tab = self.TAB_SIGNALS
            else:
                self.current_tab = self.TAB_ANOMALIES
            self.dropdown_open = False
            return
        
        # Tab-specific interactions
        content_y = self.TAB_HEIGHT + 5
        local_y = self._mouse_y - content_y
        
        if self.current_tab == self.TAB_CLUSTERS:
            self._handle_cluster_tab_click(local_y)
        elif self.current_tab == self.TAB_SIGNALS:
            self._handle_signal_tab_click(local_y)
        else:
            self._handle_anomaly_tab_click(local_y)
            
    def _handle_cluster_tab_click(self, local_y: int) -> None:
        """Handle clicks in cluster tab."""
        dropdown_y = self.PADDING
        dropdown_h = self.DROPDOWN_HEIGHT
        
        # Dropdown header click
        if dropdown_y <= local_y <= dropdown_y + dropdown_h:
            if self.PADDING <= self._mouse_x <= self.width - self.PADDING:
                self.dropdown_open = not self.dropdown_open
                return
                
        # Dropdown items click (if open)
        if self.dropdown_open and self.available_classes:
            item_y = dropdown_y + dropdown_h
            for i, cls_name in enumerate(self.available_classes):
                if item_y <= local_y <= item_y + dropdown_h:
                    self.selected_class_idx = i
                    self.dropdown_open = False
                    return
                item_y += dropdown_h
                
    def _handle_signal_tab_click(self, local_y: int) -> None:
        """Handle clicks in signal tab."""
        # 1. Handle Scrolling (apply wheel delta first)
        if self._mouse_wheel_delta != 0:
            scroll_step = 30
            self.signal_scroll_y -= self._mouse_wheel_delta * scroll_step
            # Clamp scroll
            self.signal_scroll_y = max(0, min(self.signal_scroll_y, self.signal_scroll_max))
            self._mouse_wheel_delta = 0  # Reset
            
        # 2. Handle Clicks
        # Check if click is in checkbox panel
        PANEL_WIDTH = 250
        if self._mouse_x > PANEL_WIDTH:
            return  # Clicked in graph area
            
        # Identify row
        # effective_y = local_y + self.signal_scroll_y - self.PADDING
        # row_idx = effective_y // self.ROW_HEIGHT
        
        base_classes = sorted(list({
            label.replace(" (anomaly)", "") 
            for label in self.signal_history.keys()
        }))
        
        rel_y = local_y + self.signal_scroll_y - self.PADDING
        if rel_y < 0: 
            return
            
        row_idx = rel_y // self.ROW_HEIGHT
        
        if 0 <= row_idx < len(base_classes):
            base_cls = base_classes[row_idx]
            normal_label = base_cls
            anomaly_label = f"{base_cls} (anomaly)"
            
            # Check column
            # Layout: [Name (140px)] [N (20px)] [A (20px)]
            # Box size = 16
            
            CHECK_N_X = 160
            CHECK_A_X = 200
            
            # Checkbox N
            if CHECK_N_X <= self._mouse_x <= CHECK_N_X + self.CHECKBOX_SIZE:
                if normal_label in self.signal_checkboxes:
                    self.signal_checkboxes[normal_label] = not self.signal_checkboxes[normal_label]
                    
            # Checkbox A
            if CHECK_A_X <= self._mouse_x <= CHECK_A_X + self.CHECKBOX_SIZE:
                 if anomaly_label in self.signal_checkboxes:
                    self.signal_checkboxes[anomaly_label] = not self.signal_checkboxes[anomaly_label]
                    
    def _handle_anomaly_tab_click(self, local_y: int) -> None:
        """Handle clicks and scroll in anomaly tab."""
        # 1. Scroll (apply even if not clicked)
        if self._mouse_wheel_delta != 0:
            scroll_step = 40 # Increased sensitivity
            self.anomaly_scroll_y -= self._mouse_wheel_delta * scroll_step
            self.anomaly_scroll_y = max(0, min(self.anomaly_scroll_y, self.anomaly_scroll_max))
            self._mouse_wheel_delta = 0
            
        # 2. Handle Scrollbar Dragging
        PANEL_WIDTH = 250
        DROPDOWN_H = 30
        
        # Calculate list geometry for dragging logic
        list_y_abs = self.PADDING + DROPDOWN_H + self.PADDING
        list_area_h = self.height - list_y_abs - self.TAB_HEIGHT - 5 # Approximate
        # We need exact height to map correctly. 
        # But we can assume list_area_h is roughly height - list_y_abs
        list_area_h = max(1, self.height - list_y_abs) 
        
        # If dragging
        if self.is_dragging_scrollbar:
             delta_y = self._mouse_y - self.drag_start_y
             # Map delta pixels to scroll pixels
             # Ratio: scroll_max / logic_track_h
             scroll_ratio = 1.0
             if list_area_h > 20: 
                 scroll_ratio = self.anomaly_scroll_max / list_area_h
             
             self.anomaly_scroll_y = self.drag_start_scroll_y + int(delta_y * scroll_ratio)
             self.anomaly_scroll_y = max(0, min(self.anomaly_scroll_y, self.anomaly_scroll_max))
             return

        if not self._mouse_clicked:
            return
            
        # 3. Check for Click on Scrollbar
        sb_w = 10 # Hit area wider than visual
        if PANEL_WIDTH - sb_w <= self._mouse_x <= PANEL_WIDTH:
            if list_y_abs <= local_y:
                self.is_dragging_scrollbar = True
                self.drag_start_y = self._mouse_y
                self.drag_start_scroll_y = self.anomaly_scroll_y
                return

        # 4. Handle Dropdown / Filtering
        # Left panel width is fixed (e.g. 250)
        PANEL_WIDTH = 250
        DROPDOWN_H = 30
        
        # Check if click is in dropdown area (top of panel)
        if self._mouse_x <= PANEL_WIDTH:
            # Header
            if self.PADDING <= local_y <= self.PADDING + DROPDOWN_H:
                self.anomaly_dropdown_open = not self.anomaly_dropdown_open
                return
                
            # Items (if open)
            if self.anomaly_dropdown_open:
                # Calculate filtering options
                classes = sorted(list({item['class'] for item in self.anomalies}))
                options = ["All Classes"] + classes
                
                item_y = self.PADDING + DROPDOWN_H
                for i, opt in enumerate(options):
                    if item_y <= local_y <= item_y + DROPDOWN_H:
                         if opt == "All Classes":
                             self.anomaly_filter_class = None
                         else:
                             self.anomaly_filter_class = opt
                         self.anomaly_dropdown_open = False
                         self.anomaly_scroll_y = 0 # Reset scroll
                         return
                    item_y += DROPDOWN_H
                # If clicked outside items but in panel, maybe close?
                self.anomaly_dropdown_open = False
                return

        # 3. Click on list item
        if self._mouse_x > PANEL_WIDTH:
            return
            
        # Adjust for dropdown header height (fixed) 
        # The list starts AFTER the dropdown header
        list_y_start = self.PADDING + DROPDOWN_H + self.PADDING
        
        rel_y = local_y + self.anomaly_scroll_y - list_y_start
        if rel_y < 0:
            return
            
        # Get filtered list to map index correctly
        filtered_items = [
            item for item in self.anomalies 
            if self.anomaly_filter_class is None or item['class'] == self.anomaly_filter_class
        ]
            
        ITEM_HEIGHT = 40
        row_idx = rel_y // ITEM_HEIGHT
        
        if 0 <= row_idx < len(filtered_items):
            # Find the actual index of this item in the main deque
            # We need to match by ID presumably? 
            # Or just store selected ITEM, not index
            target_item = filtered_items[row_idx]
            
            # Find index in self.anomalies
            for i, item in enumerate(self.anomalies):
                if item['id'] == target_item['id']:
                    self.selected_anomaly_idx = i
                    break
        
    def _draw_tabs(self, canvas: np.ndarray) -> None:
        """Draw tab bar."""
        tab_width = self.width // 3
        
        tabs = ["Clusters", "Signal Monitor", "Anomalies"]
        for i, name in enumerate(tabs):
            x1 = i * tab_width
            x2 = (i + 1) * tab_width
            
            # Tab background
            color = self.tab_active_color if i == self.current_tab else self.tab_color
            cv2.rectangle(canvas, (x1, 0), (x2, self.TAB_HEIGHT), color, -1)
            
            # Tab border
            cv2.rectangle(canvas, (x1, 0), (x2, self.TAB_HEIGHT), (70, 70, 70), 1)
            
            # Tab text
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            text_x = x1 + (tab_width - text_size[0]) // 2
            text_y = (self.TAB_HEIGHT + text_size[1]) // 2
            cv2.putText(canvas, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, self.text_color, 1)
                       
    def _draw_cluster_tab(
        self, 
        canvas: np.ndarray, 
        class_aware_clusterer: ClassAwareClusterer | None
    ) -> None:
        """Draw cluster visualization tab."""
        h, w = canvas.shape[:2]
        
        # Draw class dropdown
        dropdown_x = self.PADDING
        dropdown_y = self.PADDING
        dropdown_w = 180
        dropdown_h = self.DROPDOWN_HEIGHT
        
        # Dropdown header
        cv2.rectangle(canvas, (dropdown_x, dropdown_y), 
                     (dropdown_x + dropdown_w, dropdown_y + dropdown_h), 
                     self.dropdown_bg, -1)
        cv2.rectangle(canvas, (dropdown_x, dropdown_y), 
                     (dropdown_x + dropdown_w, dropdown_y + dropdown_h), 
                     (100, 100, 100), 1)
        
        # Selected class text
        selected_class = "No classes"
        if self.available_classes and 0 <= self.selected_class_idx < len(self.available_classes):
            selected_class = self.available_classes[self.selected_class_idx]
            
        cv2.putText(canvas, f"Class: {selected_class}", 
                   (dropdown_x + 8, dropdown_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
                   
        # Dropdown arrow
        arrow_x = dropdown_x + dropdown_w - 20
        arrow_y = dropdown_y + 15
        if self.dropdown_open:
            # Up arrow
            pts = np.array([[arrow_x, arrow_y + 4], [arrow_x - 5, arrow_y + 10], [arrow_x + 5, arrow_y + 10]], np.int32)
        else:
            # Down arrow
            pts = np.array([[arrow_x, arrow_y + 10], [arrow_x - 5, arrow_y + 4], [arrow_x + 5, arrow_y + 4]], np.int32)
        cv2.fillPoly(canvas, [pts], self.text_color)
        
        # Dropdown items (if open)
        if self.dropdown_open and self.available_classes:
            item_y = dropdown_y + dropdown_h
            for i, cls_name in enumerate(self.available_classes):
                bg_color = (60, 60, 60) if i == self.selected_class_idx else self.dropdown_bg
                cv2.rectangle(canvas, (dropdown_x, item_y), 
                             (dropdown_x + dropdown_w, item_y + dropdown_h), 
                             bg_color, -1)
                cv2.rectangle(canvas, (dropdown_x, item_y), 
                             (dropdown_x + dropdown_w, item_y + dropdown_h), 
                             (100, 100, 100), 1)
                             
                color = get_class_color(cls_name)
                cv2.putText(canvas, cls_name, (dropdown_x + 8, item_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                item_y += dropdown_h
                
        # Calculate plot area (avoid dropdown overlay)
        dropdown_total_h = dropdown_h
        if self.dropdown_open and self.available_classes:
            dropdown_total_h += len(self.available_classes) * dropdown_h
            
        # Draw scatter plot area
        plot_x = self.PADDING
        plot_y = dropdown_y + dropdown_total_h + self.PADDING
        plot_w = w - 2 * self.PADDING
        plot_h = h - plot_y - self.PADDING - 30  # Leave room for stats
        
        if plot_h < 50:
            return  # Not enough space
        
        # Plot border
        cv2.rectangle(canvas, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), 
                     (60, 60, 60), 1)
                     
        # Draw scatter plot if data available
        if class_aware_clusterer and selected_class != "No classes":
            self._draw_scatter_plot(canvas, class_aware_clusterer, selected_class,
                                   plot_x, plot_y, plot_w, plot_h)
        else:
            # No data message
            msg = "Waiting for clustering..."
            text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            text_x = plot_x + (plot_w - text_size[0]) // 2
            text_y = plot_y + plot_h // 2
            cv2.putText(canvas, msg, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (120, 120, 120), 1)
                       
    def _draw_scatter_plot(
        self,
        canvas: np.ndarray,
        clusterer: ClassAwareClusterer,
        class_name: str,
        x: int, y: int, w: int, h: int
    ) -> None:
        """Draw 2D scatter plot for a specific class."""
        # Get clusterer for this class
        if class_name not in clusterer._clusterers:
            return
            
        class_clusterer = clusterer._clusterers[class_name]
        result = class_clusterer.result
        
        if result is None or result.embeddings_2d is None or len(result.embeddings_2d) == 0:
            return
            
        points = result.embeddings_2d
        labels = result.labels
        
        # Normalize points to plot area
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        padding = 20
        usable_w = w - 2 * padding
        usable_h = h - 2 * padding
        
        # Downsample if too many points to avoid clutter
        MAX_POINTS = 500
        n_points = len(points)
        if n_points > MAX_POINTS:
            # Deterministic downsampling using stride to prevent "jumping"
            # Random sampling re-evaluates every frame causing flicker
            step = max(1, n_points // MAX_POINTS)
            indices = np.arange(0, n_points, step)[:MAX_POINTS]
            
            points_to_draw = points[indices]
            labels_to_draw = labels[indices]
        else:
            points_to_draw = points
            labels_to_draw = labels
        
        normalized = (points_to_draw - min_vals) / range_vals
        pixel_coords = normalized * [usable_w, usable_h]
        pixel_coords[:, 0] += x + padding
        pixel_coords[:, 1] += y + padding
        pixel_coords = pixel_coords.astype(np.int32)
        
        # Draw points
        normal_color = get_class_color(class_name, is_anomaly=False)
        anomaly_color = get_class_color(class_name, is_anomaly=True)
        
        normal_count = 0
        anomaly_count = 0
        
        for i, (px, py) in enumerate(pixel_coords):
            label = labels_to_draw[i]
            
            # Determine if this point is an anomaly (HDBSCAN noise = -1)
            is_anomaly = (label == -1)
            
            if is_anomaly:
                # Noise points = Grey and small
                # This makes them distinct from "clustered" (valid) points
                cv2.circle(canvas, (px, py), 2, (80, 80, 80), -1)
                anomaly_count += 1
            else:
                # Valid cluster points = Class color
                cv2.circle(canvas, (px, py), 2, normal_color, -1)
                normal_count += 1
                
        # Draw centroids
        for cid, info in result.clusters.items():
            # Use labels_to_draw to match pixel_coords size
            mask = labels_to_draw == cid
            if np.any(mask):
                cluster_points = pixel_coords[mask]
                cx, cy = np.mean(cluster_points, axis=0).astype(int)
                cv2.drawMarker(canvas, (cx, cy), (255, 255, 255), 
                              cv2.MARKER_CROSS, 12, 2)
                              
        # Stats text at bottom of plot
        stats_y = y + h + 20
        stats_text = f"Clusters: {result.n_clusters} | Normal: {normal_count} | Anomaly: {anomaly_count}"
        cv2.putText(canvas, stats_text, (x, stats_y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, self.text_color, 1)
                   
    def _draw_signal_tab(self, canvas: np.ndarray) -> None:
        """Draw signal monitor tab."""
        h, w = canvas.shape[:2]
        
        # Left panel: checkboxes
        checkbox_panel_w = 250
        
        # Determine base classes
        base_classes = sorted(list({
            label.replace(" (anomaly)", "") 
            for label in self.signal_history.keys()
        }))
        
        # Update scroll max
        total_list_h = len(base_classes) * self.ROW_HEIGHT + 2 * self.PADDING
        visible_h = h - self.TAB_HEIGHT
        self.signal_scroll_max = max(0, total_list_h - visible_h)
        
        # Panel background
        cv2.rectangle(canvas, (0, 0), (checkbox_panel_w, h), (40, 40, 40), -1)
        
        # Draw headers
        header_y = self.PADDING
        # (This is drawn relative to scroll? No, headers likely fixed? 
        # Actually simplest to just scroll everything or keep fixed header. 
        # Let's scroll everything for now to maximize space)
        
        row_y_start = self.PADDING - self.signal_scroll_y
        
        for i, base_cls in enumerate(base_classes):
            y = row_y_start + i * self.ROW_HEIGHT
            
            # Skip if off screen
            if y < -self.ROW_HEIGHT or y > h:
                continue
                
            # Labels
            normal_label = base_cls
            anomaly_label = f"{base_cls} (anomaly)"
            
            # Colors
            color_n = get_class_color(base_cls, False)
            color_a = get_class_color(base_cls, True)
            
            # Draw Class Name
            display_name = base_cls[:18]
            cv2.putText(canvas, display_name, (10, y + 16), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.text_color, 1)
            
            # Draw Checkbox N
            check_n = self.signal_checkboxes.get(normal_label, False)
            bx_n = 160
            cv2.rectangle(canvas, (bx_n, y+4), (bx_n+self.CHECKBOX_SIZE, y+4+self.CHECKBOX_SIZE), color_n, 1)
            if check_n:
                cv2.line(canvas, (bx_n+3, y+12), (bx_n+6, y+16), color_n, 2) # Checkmark
                cv2.line(canvas, (bx_n+6, y+16), (bx_n+13, y+8), color_n, 2)
            # Label "N" above or just implicit? implicit by column or color
            
            # Draw Checkbox A
            check_a = self.signal_checkboxes.get(anomaly_label, False)
            bx_a = 200
            cv2.rectangle(canvas, (bx_a, y+4), (bx_a+self.CHECKBOX_SIZE, y+4+self.CHECKBOX_SIZE), color_a, 1)
            if check_a:
                cv2.line(canvas, (bx_a+3, y+12), (bx_a+6, y+16), color_a, 2) # Checkmark
                cv2.line(canvas, (bx_a+6, y+16), (bx_a+13, y+8), color_a, 2)
                
        # Draw scrollbar if needed
        if self.signal_scroll_max > 0:
            sb_w = 4
            sb_x = checkbox_panel_w - sb_w - 2
            
            track_h = h - self.TAB_HEIGHT - 10
            thumb_h = max(20, int(track_h * (visible_h / total_list_h)))
            
            scroll_ratio = self.signal_scroll_y / self.signal_scroll_max
            thumb_y = self.TAB_HEIGHT + 5 + int(scroll_ratio * (track_h - thumb_h))
            
            cv2.rectangle(canvas, (sb_x, thumb_y), (sb_x+sb_w, thumb_y+thumb_h), (100, 100, 100), -1)
            
        # Right panel: graph
        graph_x = checkbox_panel_w + self.PADDING
        graph_y = self.PADDING
        graph_w = w - checkbox_panel_w - 2 * self.PADDING
        graph_h = h - 2 * self.PADDING
        
        if graph_w < 50 or graph_h < 50:
            return  # Not enough space
        
        # Graph background
        cv2.rectangle(canvas, (graph_x, graph_y), 
                     (graph_x + graph_w, graph_y + graph_h),
                     (35, 35, 35), -1)
        cv2.rectangle(canvas, (graph_x, graph_y), 
                     (graph_x + graph_w, graph_y + graph_h),
                     (60, 60, 60), 1)
                     
        # Find max value for scaling
        max_val = 1
        for label, is_checked in self.signal_checkboxes.items():
            if is_checked and label in self.signal_history:
                history = self.signal_history[label]
                if history.counts:
                     max_val = max(max_val, history.max_count)
                    
        # Draw grid lines
        for i in range(5):
            y_pos = graph_y + int((i / 4) * graph_h)
            cv2.line(canvas, (graph_x, y_pos), (graph_x + graph_w, y_pos), 
                    (50, 50, 50), 1)
            # Y-axis label
            val = int(max_val * (1 - i / 4))
            cv2.putText(canvas, str(val), (graph_x + 5, y_pos + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
                       
        # Draw signal lines
        for label, is_checked in self.signal_checkboxes.items():
            if not is_checked or label not in self.signal_history:
                continue
                
            history = self.signal_history[label]
            if len(history.counts) < 2:
                continue
                
            color = get_class_color(history.class_name, history.is_anomaly)
            
            # Build points
            counts = list(history.counts)
            n_points = len(counts)
            points = []
            
            for i, count in enumerate(counts):
                px = graph_x + int((i / max(n_points - 1, 1)) * graph_w)
                py = graph_y + graph_h - int((count / max(max_val, 1)) * graph_h)
                points.append([px, py])
                
            if len(points) >= 2:
                pts = np.array(points, dtype=np.int32)
                cv2.polylines(canvas, [pts], False, color, 2)
                
        # X-axis label
        cv2.putText(canvas, "Time (frames)", 
                   (graph_x + graph_w // 2 - 40, graph_y + graph_h - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
                   
    def show(self, class_aware_clusterer: ClassAwareClusterer | None = None) -> None:
        """Draw and display the dashboard window.
        
        Args:
            class_aware_clusterer: Clusterer for accessing per-class data
        """
        self._init_window()
        img = self.draw(class_aware_clusterer)
        cv2.imshow(self.window_name, img)
        
    def _draw_anomaly_tab(self, canvas: np.ndarray) -> None:
        """Draw anomaly review tab."""
        h, w = canvas.shape[:2]
        
        # Left Panel: List
        PANEL_WIDTH = 250
        cv2.rectangle(canvas, (0, 0), (PANEL_WIDTH, h), (40, 40, 40), -1)
        
        # Dropdown area
        dropdown_x = self.PADDING
        dropdown_y = self.PADDING
        dropdown_w = PANEL_WIDTH - 2 * self.PADDING
        dropdown_h = 30
        
        # Filter Logic
        filtered_items = [
            item for item in self.anomalies 
            if self.anomaly_filter_class is None or item['class'] == self.anomaly_filter_class
        ]
        
        # Calculate list height
        ITEM_HEIGHT = 40
        num_items = len(filtered_items)
        
        # List starts below dropdown
        list_y_abs = dropdown_y + dropdown_h + self.PADDING
        list_area_h = h - list_y_abs
        total_list_h = num_items * ITEM_HEIGHT
        
        self.anomaly_scroll_max = max(0, total_list_h - list_area_h)
        y_start_draw = list_y_abs - self.anomaly_scroll_y
        
        # Draw List Items
        for i, item in enumerate(filtered_items):
            y = y_start_draw + i * ITEM_HEIGHT
            
            # Clip
            if y < list_y_abs - ITEM_HEIGHT or y > h:
                continue
                
            # Highlight selected (check ID)
            is_selected = False
            if 0 <= self.selected_anomaly_idx < len(self.anomalies):
                 if self.anomalies[self.selected_anomaly_idx]['id'] == item['id']:
                     is_selected = True
            
            if is_selected:
                cv2.rectangle(canvas, (0, y), (PANEL_WIDTH, y + ITEM_HEIGHT), (60, 60, 60), -1)
                cv2.rectangle(canvas, (0, y), (PANEL_WIDTH, y + ITEM_HEIGHT), (100, 100, 100), 1)
            
            # Text
            time_str = item["time"].strftime("%H:%M:%S")
            label = f"{item['class']} #{item['id']}"
            color = get_class_color(item['class'], True)
            
            cv2.putText(canvas, label, (10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1)
            cv2.putText(canvas, time_str, (10, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (150, 150, 150), 1)
                       
        # Draw Dropdown (Last to overlay list)
        cv2.rectangle(canvas, (dropdown_x, dropdown_y), 
                     (dropdown_x + dropdown_w, dropdown_y + dropdown_h), 
                     self.dropdown_bg, -1)
        cv2.rectangle(canvas, (dropdown_x, dropdown_y), 
                     (dropdown_x + dropdown_w, dropdown_y + dropdown_h), 
                     (100, 100, 100), 1)
                     
        current_filter = self.anomaly_filter_class if self.anomaly_filter_class else "All Classes"
        cv2.putText(canvas, f"Filter: {current_filter}", (dropdown_x + 8, dropdown_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
                   
        # Dropdown list (if open)
        if self.anomaly_dropdown_open:
             classes = sorted(list({item['class'] for item in self.anomalies}))
             options = ["All Classes"] + classes
             
             opt_y = dropdown_y + dropdown_h
             for opt in options:
                 cv2.rectangle(canvas, (dropdown_x, opt_y), 
                              (dropdown_x + dropdown_w, opt_y + dropdown_h), 
                              self.dropdown_bg, -1)
                 cv2.rectangle(canvas, (dropdown_x, opt_y), 
                              (dropdown_x + dropdown_w, opt_y + dropdown_h), 
                              (80, 80, 80), 1)
                 cv2.putText(canvas, opt, (dropdown_x + 8, opt_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
                 opt_y += dropdown_h

        # Scrollbar (List)
        if self.anomaly_scroll_max > 0:
            sb_w = 4
            sb_x = PANEL_WIDTH - sb_w - 2
            track_h = list_area_h - 10
            scroll_ratio = self.anomaly_scroll_y / self.anomaly_scroll_max
            thumb_h = max(20, int(track_h * (list_area_h / total_list_h))) if total_list_h > 0 else 20
            thumb_y = list_y_abs + 5 + int(scroll_ratio * (track_h - thumb_h))
            
            cv2.rectangle(canvas, (sb_x, thumb_y), (sb_x+sb_w, thumb_y+thumb_h), (100, 100, 100), -1)

        # Right Panel: Preview
        preview_x = PANEL_WIDTH + self.PADDING
        preview_y = self.PADDING
        preview_w = w - preview_x - self.PADDING
        preview_h = h - 2 * self.PADDING
        
        if 0 <= self.selected_anomaly_idx < len(self.anomalies):
            item = self.anomalies[self.selected_anomaly_idx]
            img = item["image"]
            
            if img is not None and img.size > 0:
                # Resize to fit while maintaining aspect ratio
                ih, iw = img.shape[:2]
                scale = min(preview_w / iw, preview_h / ih)
                new_w = int(iw * scale)
                new_h = int(ih * scale)
                
                resized = cv2.resize(img, (new_w, new_h))
                
                # Center
                x_off = preview_x + (preview_w - new_w) // 2
                y_off = preview_y + (preview_h - new_h) // 2
                
                canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
                
                # Draw border
                cv2.rectangle(canvas, (x_off, y_off), (x_off+new_w, y_off+new_h), (255, 255, 255), 1)
                
                # Info overlay
                cv2.putText(canvas, f"Class: {item['class']}", (preview_x, preview_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 1)
                cv2.putText(canvas, f"Track ID: {item['id']}", (preview_x, preview_y + 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 1)
                cv2.putText(canvas, f"Time: {item['time'].strftime('%H:%M:%S')}", (preview_x, preview_y + 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 1)
        else:
            # No selection
            msg = "Select an anomaly"
            cv2.putText(canvas, msg, (preview_x + 50, preview_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
