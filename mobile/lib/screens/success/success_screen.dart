import 'package:flutter/material.dart';
import '../../core/theme/app_theme.dart';
import '../../models/vehicle_report.dart';

class SuccessScreen extends StatelessWidget {
  final VehicleReport report;
  const SuccessScreen({super.key, required this.report});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.check_circle, size: 84, color: AppColors.success),
              const SizedBox(height: 20),
              Text('Opened for Review on X', style: Theme.of(context).textTheme.headlineMedium, textAlign: TextAlign.center),
              const SizedBox(height: 12),
              const Text(
                'Review the pre-filled post in X and tap Post there to actually submit it. '
                'This app never publishes anything on your behalf.',
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 32),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: () => Navigator.of(context).popUntil((r) => r.isFirst),
                  child: const Text('Report Another Vehicle'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
