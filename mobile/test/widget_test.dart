import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:smoke_watch/main.dart';

void main() {
  testWidgets('App boots to home screen with Capture Vehicle button', (tester) async {
    await tester.pumpWidget(const SmokeWatchApp());

    expect(find.text('SmokeWatch'), findsOneWidget);
    expect(find.text('Detect. Verify. Report.'), findsOneWidget);
    expect(find.widgetWithText(ElevatedButton, 'Capture Vehicle'), findsOneWidget);
  });
}
