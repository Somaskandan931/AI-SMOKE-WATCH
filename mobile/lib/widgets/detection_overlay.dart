import 'dart:io';
import 'package:flutter/material.dart';
import '../models/detection_result.dart';

/// Draws vehicle/smoke bounding boxes over the captured image, per the PRD's
/// "AI RESULTS" screen spec ("Show: Original image, Bounding boxes...").
class DetectionOverlay extends StatelessWidget {
  final File image;
  final List<Detection> detections;
  final double imageWidth;
  final double imageHeight;

  const DetectionOverlay({
    super.key,
    required this.image,
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
  });

  @override
  Widget build(BuildContext context) {
    return AspectRatio(
      aspectRatio: imageWidth / imageHeight,
      child: LayoutBuilder(
        builder: (context, constraints) {
          final scaleX = constraints.maxWidth / imageWidth;
          final scaleY = constraints.maxHeight / imageHeight;
          return Stack(
            fit: StackFit.expand,
            children: [
              ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: Image.file(image, fit: BoxFit.cover),
              ),
              CustomPaint(
                painter: _BoxPainter(detections: detections, scaleX: scaleX, scaleY: scaleY),
              ),
            ],
          );
        },
      ),
    );
  }
}

class _BoxPainter extends CustomPainter {
  final List<Detection> detections;
  final double scaleX, scaleY;

  _BoxPainter({required this.detections, required this.scaleX, required this.scaleY});

  @override
  void paint(Canvas canvas, Size size) {
    for (final d in detections) {
      final isSmoke = d.label == 'exhaust_smoke';
      final paint = Paint()
        ..color = isSmoke ? Colors.orangeAccent : Colors.greenAccent
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3;

      final rect = Rect.fromLTRB(
        d.box.x1 * scaleX,
        d.box.y1 * scaleY,
        d.box.x2 * scaleX,
        d.box.y2 * scaleY,
      );
      canvas.drawRRect(RRect.fromRectAndRadius(rect, const Radius.circular(6)), paint);

      final label = '${d.label} ${(d.confidence * 100).toStringAsFixed(0)}%';
      final textPainter = TextPainter(
        text: TextSpan(
          text: label,
          style: TextStyle(
            color: Colors.white,
            fontSize: 12,
            backgroundColor: isSmoke ? Colors.orangeAccent.shade700 : Colors.green.shade800,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      textPainter.paint(canvas, Offset(rect.left, rect.top - textPainter.height - 2));
    }
  }

  @override
  bool shouldRepaint(covariant _BoxPainter oldDelegate) => true;
}
