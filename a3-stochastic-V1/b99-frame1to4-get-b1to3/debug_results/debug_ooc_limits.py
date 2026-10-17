"""
Debug instrumentation helper for ooc_l/ooc_r limit tracing.

HOW TO USE:
-----------
1. In m1_vertexMove_Va001_env.py, at the TOP of the file, add:
       from debug_ooc_limits import DebugOOCLimits
       ooc_debug = DebugOOCLimits()

2. In apply_smooth_limits_to_actions() (line ~7190), AFTER computing t_limit_ooc_l/r,
   add these 3 lines:

       ooc_debug.log_clipping(
           step=self.step_counter_internal,
           unit_prime_ooc_l=self.UNIT_PRIME_OOC_L,
           unit_prime_ooc_r=self.UNIT_PRIME_OOC_R,
           unit_prime_oor=self.UNIT_PRIME_OOR,
           t_limit_ooc_l=t_limit_ooc_l,
           t_limit_ooc_r=t_limit_ooc_r,
           t_limit_oor=t_limit_oor,
           action_ooc_l_before=action[start_ooc_l:end_ooc_l].copy(),
           action_ooc_r_before=action[start_ooc_r:end_ooc_r].copy(),
           action_oor_before=action[0:len_oor].copy(),
       )

3. AFTER the clipping lines (after action[start_ooc_r:end_ooc_r] = action_ooc_r), add:

       ooc_debug.log_after_clipping(
           action_ooc_l_after=action[start_ooc_l:end_ooc_l].copy(),
           action_ooc_r_after=action[start_ooc_r:end_ooc_r].copy(),
           action_oor_after=action[0:len_oor].copy(),
       )

4. Make sure the sys.path includes the debug_results folder. In d1-adhoc-for-b99.py add:
       sys.path.insert(0, r"C:\\Users\\zpa18\\a3-stochastic-V1\\b99-frame1to4-get-b1to3\\debug_results")

The debug output will be written to:
  C:\\Users\\zpa18\\a3-stochastic-V1\\b99-frame1to4-get-b1to3\\debug_results\\ooc_limits_trace.txt
"""

import os
import numpy as np
import datetime
from paths import b99_frame1to4_path

class DebugOOCLimits:
    def __init__(self, output_dir=str(b99_frame1to4_path / "debug_results")):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        self.filepath = os.path.join(self.output_dir, f"ooc_limits_trace_{timestamp}.txt")
        self.log_every = 1000  # Log every N steps to avoid huge files
        self._pending = {}

        # Write header
        with open(self.filepath, 'w') as f:
            f.write(f"=== OOC Limits Debug Trace ===\n")
            f.write(f"Started: {timestamp}\n")
            f.write(f"Logging every {self.log_every} steps\n")
            f.write(f"{'='*80}\n\n")

    def log_clipping(self, step, unit_prime_ooc_l, unit_prime_ooc_r, unit_prime_oor,
                     t_limit_ooc_l, t_limit_ooc_r, t_limit_oor,
                     action_ooc_l_before, action_ooc_r_before, action_oor_before):
        """Call BEFORE clipping is applied."""
        self._pending = {
            'step': step,
            'unit_prime_ooc_l': unit_prime_ooc_l,
            'unit_prime_ooc_r': unit_prime_ooc_r,
            'unit_prime_oor': unit_prime_oor,
            't_limit_ooc_l': t_limit_ooc_l,
            't_limit_ooc_r': t_limit_ooc_r,
            't_limit_oor': t_limit_oor,
            'action_ooc_l_before': action_ooc_l_before,
            'action_ooc_r_before': action_ooc_r_before,
            'action_oor_before': action_oor_before,
        }

    def log_after_clipping(self, action_ooc_l_after, action_ooc_r_after, action_oor_after):
        """Call AFTER clipping is applied."""
        if not self._pending:
            return

        step = self._pending['step']
        if step % self.log_every != 0:
            self._pending = {}
            return

        p = self._pending

        # Calculate how much was clipped
        ooc_l_before_mag = np.linalg.norm(p['action_ooc_l_before'])
        ooc_l_after_mag = np.linalg.norm(action_ooc_l_after)
        ooc_l_clipped_pct = (1.0 - ooc_l_after_mag / max(ooc_l_before_mag, 1e-15)) * 100

        ooc_r_before_mag = np.linalg.norm(p['action_ooc_r_before'])
        ooc_r_after_mag = np.linalg.norm(action_ooc_r_after)
        ooc_r_clipped_pct = (1.0 - ooc_r_after_mag / max(ooc_r_before_mag, 1e-15)) * 100

        oor_before_mag = np.linalg.norm(p['action_oor_before'])
        oor_after_mag = np.linalg.norm(action_oor_after)
        oor_clipped_pct = (1.0 - oor_after_mag / max(oor_before_mag, 1e-15)) * 100

        with open(self.filepath, 'a') as f:
            f.write(f"--- Step {step} ---\n")
            f.write(f"  UNIT_PRIME_OOR:   {p['unit_prime_oor']:.8e}   t_limit_oor:   {p['t_limit_oor']:.8e}\n")
            f.write(f"  UNIT_PRIME_OOC_L: {p['unit_prime_ooc_l']:.8e}   t_limit_ooc_l: {p['t_limit_ooc_l']:.8e}\n")
            f.write(f"  UNIT_PRIME_OOC_R: {p['unit_prime_ooc_r']:.8e}   t_limit_ooc_r: {p['t_limit_ooc_r']:.8e}\n")
            f.write(f"\n")
            f.write(f"  OOR   gradient magnitude: before={oor_before_mag:.8e}  after={oor_after_mag:.8e}  clipped={oor_clipped_pct:.1f}%\n")
            f.write(f"  OOC_L gradient magnitude: before={ooc_l_before_mag:.8e}  after={ooc_l_after_mag:.8e}  clipped={ooc_l_clipped_pct:.1f}%\n")
            f.write(f"  OOC_R gradient magnitude: before={ooc_r_before_mag:.8e}  after={ooc_r_after_mag:.8e}  clipped={ooc_r_clipped_pct:.1f}%\n")
            f.write(f"\n")

            # Flag if the clipping is severe (>50%)
            if ooc_l_clipped_pct > 50:
                f.write(f"  *** WARNING: OOC_L gradient is {ooc_l_clipped_pct:.0f}% clipped! Limit too small. ***\n")
            if ooc_r_clipped_pct > 50:
                f.write(f"  *** WARNING: OOC_R gradient is {ooc_r_clipped_pct:.0f}% clipped! Limit too small. ***\n")

            # Ratio comparison
            if p['t_limit_oor'] > 1e-15:
                ratio_l = p['t_limit_ooc_l'] / p['t_limit_oor']
                ratio_r = p['t_limit_ooc_r'] / p['t_limit_oor']
                f.write(f"  Limit ratio OOC_L/OOR: {ratio_l:.4f}   OOC_R/OOR: {ratio_r:.4f}\n")

            f.write(f"\n")

        self._pending = {}
