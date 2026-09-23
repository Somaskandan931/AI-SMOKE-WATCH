import 'package:geolocator/geolocator.dart';

class LocationResult {
  final double? latitude;
  final double? longitude;
  final String label;
  final bool permissionGranted;

  const LocationResult({
    required this.latitude,
    required this.longitude,
    required this.label,
    required this.permissionGranted,
  });

  static const denied = LocationResult(
    latitude: null,
    longitude: null,
    label: 'Location unavailable',
    permissionGranted: false,
  );
}

/// FR-11: "The application shall capture the user's location only after
/// obtaining the required permission." Never collects location silently.
class LocationService {
  Future<LocationResult> getApproximateLocation() async {
    final serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      return LocationResult.denied;
    }

    var permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
    }

    if (permission == LocationPermission.denied || permission == LocationPermission.deniedForever) {
      return LocationResult.denied;
    }

    try {
      final position = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(accuracy: LocationAccuracy.medium),
      );
      // Reverse geocoding is intentionally left out of the MVP — a
      // lat/lng pair plus city-level label the user can edit is enough
      // for the report and keeps the app free of another API dependency.
      return LocationResult(
        latitude: position.latitude,
        longitude: position.longitude,
        label: '${position.latitude.toStringAsFixed(4)}, ${position.longitude.toStringAsFixed(4)}',
        permissionGranted: true,
      );
    } catch (_) {
      return LocationResult.denied;
    }
  }
}
