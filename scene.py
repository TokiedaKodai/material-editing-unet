scene_xml = \
'''<?xml version="1.0" encoding="utf-8"?>

<scene version="0.5.0">
	<integrator type="photonmapper">
        <integer name="directSamples" value="32"/>
        <integer name="glossySamples" value="512"/>
    </integrator>

	<sensor type="perspective">
		<transform name="toWorld">
			<lookAt origin="0, 0, 5" target="0, 0, 0" up="0, 1, 0"/>
		</transform>
	</sensor>



	<bsdf type="diffuse" id="diffuse">
        <spectrum name="reflectance" value="1"/>
    </bsdf>

    <bsdf type="roughconductor" id="cu">
        <string name="material" value="Cu"/>
        <float name="alpha" value="0.3"/>
    </bsdf>
	<bsdf type="roughconductor" id="cu2o">
        <string name="material" value="Cu2O"/>
        <float name="alpha" value="0.3"/>
    </bsdf>
	<bsdf type="roughconductor" id="cuo">
        <string name="material" value="CuO"/>
        <float name="alpha" value="0.3"/>
    </bsdf>
    <bsdf type="roughconductor" id="au">
        <string name="material" value="Au"/>
        <float name="alpha" value="0.3"/>
    </bsdf>
    <bsdf type="roughconductor" id="carbon">
        <string name="material" value="a-C"/>
        <float name="alpha" value="0.2"/>
    </bsdf>

    <bsdf type="roughplastic" id="plastic"/>


	<shape type="obj">
		<string name="filename" value="%s"/>
		<ref id="%s"/>
	</shape>


	<emitter type="directional">
		<spectrum name="irradiance" value="100"/>
		<vector name="direction" x="0.02" y="0" z="-1"/>
	</emitter>


</scene>
'''

'''
	<emitter type="directional">
		<spectrum name="irradiance" value="2"/>
		<vector name="direction" x="0.02" y="0" z="-1"/>
	</emitter>

	<emitter type="spot">
		<transform name="toWorld">
			<lookAt origin="-0.1, 0, 5" target="0, 0, 0"/>
		</transform>
        <spectrum name="intensity" value="30"/>
        <float name="cutoffAngle" value="45"/>
	</emitter>

    <shape type="sphere">
        <point name="center" x="-0.2" y="0" z="5"/>
        <float name="radius" value="0.1"/>
        <emitter type="area">
            <spectrum name="radiance" value="2000"/>
        </emitter>
    </shape>

	<emitter type="envmap" id="Area_002-light">
		<string name="filename" value="envmap.exr"/>
		<transform name="toWorld">
			<rotate y="1" angle="-180"/>
			<matrix value="-0.224951 -0.000001 -0.974370 0.000000 -0.974370 0.000000 0.224951 0.000000 0.000000 1.000000 -0.000001 8.870000 0.000000 0.000000 0.000000 1.000000 "/>
		</transform>
		<float name="scale" value="3"/>
	</emitter>

    <bsdf type="roughconductor" id="au">
        <string name="material" value="Au"/>
        <float name="alpha" value="0.3"/>
    </bsdf>
'''