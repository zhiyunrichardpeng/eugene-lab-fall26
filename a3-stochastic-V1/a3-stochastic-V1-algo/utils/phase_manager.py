class PhaseManager:
    def __init__(self, max_phases):
        self.current_phase = 1
        self.max_phases = max_phases
        self.phase_data = {
            phase: {
                'gradient_magnitudes': {},
                'term_history': {
                    'steps': [],
                    'distance': [],
                    'strain': [],
                    'muscle': []
                },
                'logs': {
                    'distance': [],
                    'step_counter': [],
                    'energy': [],
                    # ... other logs
                }
            }
            for phase in range(1, max_phases+1)
        }
        self.phase_params = {}  # Stores phase-specific parameters