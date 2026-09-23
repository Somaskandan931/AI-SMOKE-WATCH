import 'package:flutter/material.dart';
import '../../core/theme/app_theme.dart';
import '../../models/vehicle_report.dart';
import '../vehicle_capture/vehicle_capture_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            children: [
              const Spacer(flex: 2),
              const Icon(Icons.air, size: 72, color: AppColors.primary),
              const SizedBox(height: 20),
              Text('SMOKE WATCH',
                  style: Theme.of(context).textTheme.headlineMedium?.copyWith(letterSpacing: 1.5)),
              const SizedBox(height: 8),
              Text(
                'Detect. Verify. Report.',
                style: Theme.of(context).textTheme.titleLarge?.copyWith(color: AppColors.primaryLight),
              ),
              const SizedBox(height: 16),
              Text(
                'Help report vehicles emitting visible exhaust smoke. '
                'Our AI verifies visible smoke in your photo before a report can be sent.',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodyMedium,
              ),
              const Spacer(flex: 3),
              ElevatedButton.icon(
                icon: const Icon(Icons.camera_alt),
                label: const Text('Capture Vehicle'),
                onPressed: () => Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => VehicleCaptureScreen(report: VehicleReport())),
                ),
              ),
              const SizedBox(height: 20),
              Text(
                'The AI detects visible smoke only. It does not measure pollutant '
                'levels or confirm a legal emission violation.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 11, color: AppColors.textSecondary),
              ),
              const SizedBox(height: 8),
            ],
          ),
        ),
      ),
    );
  }
}
