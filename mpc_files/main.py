import ...

while True:
	depth = ...
	pos = ...
	vel = ...
	att = ...
	omega = ...

	x, y = next_target(depth)

	x, y, z = step((x, y))

	actual_mpc(x, y, z, pos, vel, att, omega)