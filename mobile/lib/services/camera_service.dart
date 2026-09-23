import 'package:image_picker/image_picker.dart';

/// FR-01 / FR-02: capture from camera or select from gallery.
///
/// Returns XFile (not dart:io File) — File has no working implementation
/// on Flutter Web (image_picker gives back a blob: URL there, not a real
/// filesystem path), while XFile.readAsBytes() works identically on both
/// web and native.
class CameraService {
  final ImagePicker _picker = ImagePicker();

  Future<XFile?> captureFromCamera({int imageQuality = 85}) {
    return _picker.pickImage(
      source: ImageSource.camera,
      imageQuality: imageQuality,
      preferredCameraDevice: CameraDevice.rear,
    );
  }

  Future<XFile?> pickFromGallery({int imageQuality = 85}) {
    return _picker.pickImage(
      source: ImageSource.gallery,
      imageQuality: imageQuality,
    );
  }
}