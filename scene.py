scene_xml = \
'''<?xml version="1.0" encoding="utf-8"?>

<scene version="0.5.0">
	<integrator type="path"/>

	<sensor type="perspective">
		<transform name="toWorld">
			<lookAt origin="0, 0, 5" target="0, 0, 0" up="0, 1, 0"/>
		</transform>
	</sensor>



	<bsdf type="diffuse" id="diffuse">
        <spectrum name="reflectance" value="10"/>
    </bsdf>

    <bsdf type="roughdielectric" id="dielectric">
        <string name="intIOR" value="water"/>
        <string name="extIOR" value="air"/>
    </bsdf>

    <bsdf type="roughconductor" id="cu">
        <string name="material" value="Cu"/>
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



	<emitter type="directional">
		<spectrum name="irradiance" value="2"/>
		<vector name="direction" x="0.02" y="0" z="-1"/>
	</emitter>

	<!-- <emitter type="spot">
		<transform name="toWorld">
			<lookAt origin="-0.02, 0, 0" target="0, 0, -1"/>
		</transform>
	</emitter> -->


	<shape type="obj">
		<string name="filename" value="%s"/>
		<ref id="%s"/>
	</shape>


</scene>
'''