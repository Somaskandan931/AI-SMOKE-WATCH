class AppConstants {
  /// Point this at your FastAPI backend.
  /// - Android emulator -> host machine: http://10.0.2.2:8000
  /// - iOS simulator -> host machine: http://localhost:8000
  /// - Physical device -> your machine's LAN IP, e.g. http://192.168.1.20:8000
  static const String apiBaseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'http://10.0.2.2:8000/api',
  );

  static const String authorityDisplayName = 'Chennai City Traffic Police';

  static const Duration requestTimeout = Duration(seconds: 30);
}
