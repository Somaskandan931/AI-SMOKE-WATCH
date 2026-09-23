import 'package:flutter/material.dart';
import '../core/theme/app_theme.dart';

/// PRD section 11 / Screen 5: shows registration, location, date/time,
/// smoke result, and the generated report text before the user shares it.
class ReportCard extends StatelessWidget {
  final String registration;
  final String location;
  final String timestamp;
  final bool smokeDetected;
  final String reportText;

  const ReportCard({
    super.key,
    required this.registration,
    required this.location,
    required this.timestamp,
    required this.smokeDetected,
    required this.reportText,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _field('Registration', registration),
            const SizedBox(height: 14),
            _field('Location', location),
            const SizedBox(height: 14),
            _field('Date & Time', timestamp),
            const SizedBox(height: 14),
            Row(
              children: [
                Icon(smokeDetected ? Icons.check_circle : Icons.cancel,
                    color: smokeDetected ? AppColors.success : AppColors.danger, size: 18),
                const SizedBox(width: 8),
                Text(smokeDetected ? 'Visible smoke detected' : 'Smoke not detected'),
              ],
            ),
            const SizedBox(height: 18),
            const Divider(),
            const SizedBox(height: 10),
            Text('Generated Report', style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 8),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: AppColors.background,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(reportText, style: const TextStyle(height: 1.5)),
            ),
          ],
        ),
      ),
    );
  }

  Widget _field(String label, String value) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(
          width: 100,
          child: Text(label, style: const TextStyle(color: AppColors.textSecondary)),
        ),
        Expanded(child: Text(value, style: const TextStyle(fontWeight: FontWeight.w600))),
      ],
    );
  }
}
