import 'package:flutter/material.dart';
import '../core/theme/app_theme.dart';
import '../models/detection_result.dart';

/// PRD section 7/Step 3-5 + "AI RESULTS" mobile UI spec: shows vehicle %,
/// smoke % (or "Not Detected"), and the final reporting-eligible decision.
class SmokeStatusCard extends StatelessWidget {
  final DetectionResult result;
  const SmokeStatusCard({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _row('Vehicle', result.vehicleDetected, result.vehicleConfidence),
            const Divider(height: 28),
            _row('Visible Smoke', result.smokeDetected, result.smokeConfidence),
            const SizedBox(height: 20),
            _verdict(context),
            if (result.mode == 'mock') ...[
              const SizedBox(height: 12),
              Text(
                'Running in demo mode (no trained model loaded yet) — '
                'results come from a placeholder heuristic, not a trained YOLO model.',
                style: TextStyle(fontSize: 12, color: AppColors.textSecondary, fontStyle: FontStyle.italic),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _row(String label, bool detected, double confidence) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(label, style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
        Text(
          detected ? '${(confidence * 100).toStringAsFixed(0)}%' : 'Not Detected',
          style: TextStyle(
            fontWeight: FontWeight.w700,
            fontSize: 16,
            color: detected ? AppColors.success : AppColors.textSecondary,
          ),
        ),
      ],
    );
  }

  Widget _verdict(BuildContext context) {
    final eligible = result.reportingAllowed;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 16),
      decoration: BoxDecoration(
        color: eligible ? AppColors.success.withOpacity(0.12) : AppColors.danger.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Icon(eligible ? Icons.check_circle : Icons.cancel, color: eligible ? AppColors.success : AppColors.danger),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              eligible ? 'Reporting Eligible' : 'Reporting Disabled',
              style: TextStyle(
                fontWeight: FontWeight.w700,
                color: eligible ? AppColors.success : AppColors.danger,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
