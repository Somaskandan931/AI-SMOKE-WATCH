import 'dart:io';

import 'package:image_picker/image_picker.dart';

/// FR-01/FR-02: capture from camera or pick from gallery.
class CameraService {
  final ImagePicker _picker = ImagePicker();

  Future<File?> captureFromCamera() async {
    final XFile? file = await _picker.pickImage(
      source: ImageSource.camera,
      imageQuality: 90,
      maxWidth: 1600,
    );
    return file == null ? null : File(file.path);
  }

  Future<File?> pickFromGallery() async {
    final XFile? file = await _picker.pickImage(
      source: ImageSource.gallery,
      imageQuality: 90,
      maxWidth: 1600,
    );
    return file == null ? null : File(file.path);
  }
}
