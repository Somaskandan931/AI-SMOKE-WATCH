import 'package:flutter/material.dart';
import '../core/theme/app_theme.dart';

class LoadingIndicator extends StatelessWidget {
  final String label;
  const LoadingIndicator({super.key, required this.label});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const CircularProgressIndicator(color: AppColors.primary),
          const SizedBox(height: 16),
          Text(label, style: Theme.of(context).textTheme.bodyMedium),
        ],
      ),
    );
  }
}
