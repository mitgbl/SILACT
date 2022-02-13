% imwritestack(stack, filename)
%
% Save image stack to multi-page uncompressed 32-bit floating-point TIFF file.
%
% stack: the 3D stack to be saved.
% filename: file name.
%
% (c) Cedric Vonesch, Biomedical Imaging Group, EPFL

function imwritestack(stack, filename)
	t = Tiff(filename, 'w');

	tagstruct.ImageLength = size(stack, 1);
	tagstruct.ImageWidth = size(stack, 2);
	tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
	tagstruct.BitsPerSample = 32;
	tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
	tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;

	for k = 1:size(stack, 3)
		t.setTag(tagstruct)
		t.write(single(stack(:, :, k)));
		t.writeDirectory();
	end

	t.close();
end