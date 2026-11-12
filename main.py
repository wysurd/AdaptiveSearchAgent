import csv
import random
from datetime import datetime, timezone
from pathlib import Path

import pygame

from belief import Belief
from pathfinding import astar

pygame.init()

#map settings
TILE_SIZE = 40

game_map = [
    "111111111111111111111111",
    "100000010000000010000001",
    "100000010000000010000001",
    "100000010000000010000001",
    "100000000000000000000001",
    "100000010000000010000001",
    "100000010000000010000001",
    "111011111110111111110111",
    "100000000000000000000001",
    "111110111111110111011111",
    "100000010000000010000001",
    "100000010000000010000001",
    "100000000000000000000001",
    "100000010000000010000001",
    "100000010000000010000001",
    "111111111111111111111111"
]

#window settings
WIDTH = len(game_map[0]) * TILE_SIZE
HEIGHT = len(game_map) * TILE_SIZE
FPS = 60

FLOOR_COLOR = (18, 20, 24)
WALL_COLOR = (65, 69, 75)
PLAYER_COLOR = (185, 205, 190)
ENEMY_COLOR = (180, 65, 65)
PATH_COLOR = (100, 45, 45)
ORB_COLOR = (100, 190, 240)
TEXT_COLOR = (235, 235, 235)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Horror Enemy AI")

clock = pygame.time.Clock()
font = pygame.font.Font(None, 22)
large_font = pygame.font.Font(None, 54)

darkness = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

#character settings
walk_speed = 220
sprint_speed = 290
sneak_speed = 120
enemy_speed = 240
vision_range = 240

player_rect = pygame.Rect(0, 0, 24, 24)
enemy_rect = pygame.Rect(0, 0, 24, 24)

#stamina settings
max_stamina = 100
stamina_drain = 25
stamina_recovery = 18
sprint_recovery_threshold = 25

#search settings
prediction_interval = 0.25
search_duration = 12.0
commitment_duration = 1.0

search_modes = ["hybrid", "probabilistic", "baseline"]
search_mode = "hybrid"

patrol_points = [
    (20, 13),
    (3, 12),
    (3, 2),
    (20, 2)
]

#display settings
limited_visibility = False
player_view_radius = 130
show_heatmap = False
show_path = False

#objective settings
orb_radius = 10
orb_target = 20

#round settings
round_seed = 42
results_file = Path(__file__).resolve().parent / "session_results.csv"


def pixel_to_tile(position):
    #convert pixels to grid coordinates
    return (
        int(position.x // TILE_SIZE),
        int(position.y // TILE_SIZE)
    )


def tile_to_pixel(tile):
    #find the tile center
    column, row = tile

    return pygame.Vector2(
        column * TILE_SIZE + TILE_SIZE / 2,
        row * TILE_SIZE + TILE_SIZE / 2
    )


def can_see_position(position):
    #check enemy distance and blocking walls
    if enemy_position.distance_to(position) > vision_range:
        return False

    start = (round(enemy_position.x), round(enemy_position.y))
    end = (round(position.x), round(position.y))

    return not any(
        wall.clipline(start, end) for wall in walls
    )


def move_player(movement):
    #move horizontally
    player_position.x += movement.x
    player_rect.centerx = round(player_position.x)

    for wall in walls:
        if player_rect.colliderect(wall):
            if movement.x > 0:
                player_rect.right = wall.left
            elif movement.x < 0:
                player_rect.left = wall.right

            player_position.x = player_rect.centerx

    #move vertically
    player_position.y += movement.y
    player_rect.centery = round(player_position.y)

    for wall in walls:
        if player_rect.colliderect(wall):
            if movement.y > 0:
                player_rect.bottom = wall.top
            elif movement.y < 0:
                player_rect.top = wall.bottom

            player_position.y = player_rect.centery


def spawn_orb():
    #use a repeatable sequence of floor tiles
    candidates = [
        tile for tile in floor_tiles
        if tile != orb_tile
        and tile != (2, 2)
        and tile != (20, 13)
    ]

    return orb_random.choice(candidates)


def plan_path():
    #plan from the current tile center
    path = astar(
        game_map,
        pixel_to_tile(enemy_position),
        enemy_goal
    )

    return path[1:] if path is not None else []


def finish_round(outcome):
    global game_status, log_message

    #record each completed round once
    if game_status != "playing":
        return

    game_status = outcome

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": round_seed,
        "mode": search_mode,
        "limited_visibility_at_end": int(limited_visibility),
        "visibility_toggles": visibility_toggles,
        "outcome": outcome,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "orbs_collected": collected_orbs,
        "orb_target": orb_target,
        "sightings": sightings,
        "reacquisitions": reacquisitions,
        "heard_footsteps": heard_footsteps,
        "sprint_seconds": round(sprint_seconds, 2),
        "sneak_seconds": round(sneak_seconds, 2),
        "enemy_speed": enemy_speed,
        "walk_speed": walk_speed,
        "sprint_speed": sprint_speed,
        "sneak_speed": sneak_speed
    }

    try:
        needs_header = (
            not results_file.exists()
            or results_file.stat().st_size == 0
        )

        with results_file.open(
            "a",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(result.keys())
            )

            if needs_header:
                writer.writeheader()

            writer.writerow(result)

        log_message = "round saved to session_results.csv"

    except OSError as error:
        log_message = "could not save round results"
        print(f"{log_message}: {error}")

    print(
        f"{outcome} | mode: {search_mode} | "
        f"seed: {round_seed} | orbs: {collected_orbs} | "
        f"time: {elapsed_seconds:.2f}s"
    )


def reset_game():
    global player_position, enemy_position
    global enemy_path, enemy_goal, enemy_state
    global patrol_index, last_seen_tile, search_anchor
    global search_timer, prediction_timer
    global checked_anchor, committed_goal, commitment_timer
    global collected_orbs, orb_tile, game_status
    global stamina, exhausted, movement_mode, footstep_distance
    global elapsed_seconds, sprint_seconds, sneak_seconds
    global sightings, reacquisitions, heard_footsteps
    global was_visible, has_been_seen, heard_flash
    global orb_random, sound_random, log_message
    global visibility_toggles

    #separate objective and sensory randomness
    orb_random = random.Random(round_seed)
    sound_random = random.Random(round_seed + 100000)

    #reset characters
    player_position = tile_to_pixel((2, 2))
    enemy_position = tile_to_pixel((20, 13))

    player_rect.center = (
        round(player_position.x),
        round(player_position.y)
    )

    enemy_rect.center = (
        round(enemy_position.x),
        round(enemy_position.y)
    )

    #reset enemy memory
    enemy_path = []
    enemy_goal = None
    enemy_state = "patrol"
    patrol_index = 0
    last_seen_tile = None
    search_anchor = None
    search_timer = 0
    prediction_timer = 0
    checked_anchor = False
    committed_goal = None
    commitment_timer = 0
    belief.reset()

    #reset player resources
    stamina = max_stamina
    exhausted = False
    movement_mode = "walk"
    footstep_distance = 0

    #reset round measurements
    elapsed_seconds = 0
    sprint_seconds = 0
    sneak_seconds = 0
    sightings = 0
    reacquisitions = 0
    heard_footsteps = 0
    visibility_toggles = 0
    was_visible = False
    has_been_seen = False
    heard_flash = 0
    log_message = ""

    #reset objective
    collected_orbs = 0
    orb_tile = None
    orb_tile = spawn_orb()
    game_status = "playing"


def draw_orb_hint():
    #point toward an orb outside the view
    if orb_tile is None:
        return

    offset = tile_to_pixel(orb_tile) - player_position

    if offset.length() <= player_view_radius - orb_radius:
        return

    direction = offset.normalize()
    sideways = pygame.Vector2(-direction.y, direction.x)

    tip = (
        player_position
        + direction * (player_view_radius - 18)
    )

    base = tip - direction * 13

    points = [
        tip,
        base + sideways * 6,
        base - sideways * 6
    ]

    pygame.draw.polygon(
        screen,
        ORB_COLOR,
        [(round(point.x), round(point.y)) for point in points]
    )


#build the map
walls = []
floor_tiles = []

for row_index, row in enumerate(game_map):
    for column_index, tile in enumerate(row):
        if tile == "1":
            walls.append(
                pygame.Rect(
                    column_index * TILE_SIZE,
                    row_index * TILE_SIZE,
                    TILE_SIZE,
                    TILE_SIZE
                )
            )
        else:
            floor_tiles.append((column_index, row_index))

belief = Belief(game_map)

reset_game()
running = True

while running:
    #limit elapsed time
    dt = min(clock.tick(FPS) / 1000, 0.05)

    #handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            elif event.key == pygame.K_r:
                reset_game()

            elif event.key == pygame.K_n:
                round_seed = random.SystemRandom().randrange(
                    1, 1000000
                )
                reset_game()

            elif event.key == pygame.K_v:
                limited_visibility = not limited_visibility

                if game_status == "playing":
                    visibility_toggles += 1

            elif event.key == pygame.K_h:
                show_heatmap = not show_heatmap

            elif event.key == pygame.K_p:
                show_path = not show_path

            elif event.key == pygame.K_b:
                index = search_modes.index(search_mode)
                search_mode = search_modes[
                    (index + 1) % len(search_modes)
                ]
                reset_game()

    if not running:
        break

    if game_status == "playing":
        elapsed_seconds += dt
        heard_flash = max(0, heard_flash - dt)

        #read player input
        keys = pygame.key.get_pressed()

        direction = pygame.Vector2(
            int(keys[pygame.K_d] or keys[pygame.K_RIGHT])
            - int(keys[pygame.K_a] or keys[pygame.K_LEFT]),
            int(keys[pygame.K_s] or keys[pygame.K_DOWN])
            - int(keys[pygame.K_w] or keys[pygame.K_UP])
        )

        moving = direction.length_squared() > 0

        if moving:
            direction = direction.normalize()

        shift_pressed = (
            keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        )

        control_pressed = (
            keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]
        )

        if exhausted and stamina >= sprint_recovery_threshold:
            exhausted = False

        #choose movement speed
        if moving and control_pressed:
            movement_mode = "sneak"
            speed = sneak_speed
            sneak_seconds += dt

        elif (
            moving
            and shift_pressed
            and not exhausted
            and stamina > 0
        ):
            movement_mode = "sprint"
            speed = sprint_speed
            sprint_seconds += dt

        else:
            movement_mode = "walk"
            speed = walk_speed

        #update stamina
        if movement_mode == "sprint":
            stamina = max(0, stamina - stamina_drain * dt)

            if stamina == 0:
                exhausted = True
        else:
            stamina = min(
                max_stamina,
                stamina + stamina_recovery * dt
            )

        #move and measure actual travel
        previous_position = player_position.copy()
        move_player(direction * speed * dt)

        traveled = player_position.distance_to(previous_position)
        heard_tile = None

        #generate footsteps from actual movement
        if movement_mode == "sneak":
            footstep_distance = 0

        else:
            footstep_distance += traveled

            if footstep_distance >= 80:
                footstep_distance %= 80

                hearing_range = (
                    300 if movement_mode == "sprint" else 110
                )

                if (
                    enemy_position.distance_to(player_position)
                    <= hearing_range
                ):
                    #report an approximate sensory location
                    actual_tile = pixel_to_tile(player_position)

                    possible_reports = [
                        actual_tile,
                        *belief.neighbors[actual_tile]
                    ]

                    heard_tile = sound_random.choice(
                        possible_reports
                    )

                    heard_footsteps += 1
                    heard_flash = 0.7

        #update direct vision
        player_visible = can_see_position(player_position)

        if player_visible:
            if not was_visible:
                sightings += 1

                if has_been_seen:
                    reacquisitions += 1

                has_been_seen = True

            enemy_state = "chase"
            last_seen_tile = pixel_to_tile(player_position)
            belief.observe_player(last_seen_tile)
            search_timer = 0
            prediction_timer = 0

        elif enemy_state == "chase":
            #search from the last sighting
            enemy_state = "search"
            search_anchor = last_seen_tile
            checked_anchor = False
            committed_goal = None
            commitment_timer = 0
            search_timer = 0
            prediction_timer = 0

            if search_mode != "baseline":
                belief.predict()

        was_visible = player_visible

        #use sound when direct sight is unavailable
        if heard_tile is not None and not player_visible:
            enemy_state = "search"
            search_timer = 0
            search_anchor = heard_tile

            if search_mode == "baseline":
                checked_anchor = False

            else:
                belief.observe_sound(heard_tile)

                #sound can redirect a search immediately
                checked_anchor = True
                committed_goal = None
                commitment_timer = 0

        #update search belief
        if enemy_state == "search":
            search_timer += dt

            commitment_timer = max(
                0,
                commitment_timer - dt
            )

            if search_mode != "baseline":
                prediction_timer += dt

                while prediction_timer >= prediction_interval:
                    belief.predict()
                    prediction_timer -= prediction_interval

                visible_tiles = [
                    tile for tile in floor_tiles
                    if can_see_position(tile_to_pixel(tile))
                ]

                belief.observe_empty(visible_tiles)

            #use an equal inactivity budget for all modes
            if search_timer >= search_duration:
                enemy_state = "patrol"

        #choose a destination
        enemy_tile = pixel_to_tile(enemy_position)

        if enemy_state == "patrol":
            if not enemy_path:
                if enemy_tile == patrol_points[patrol_index]:
                    patrol_index = (
                        patrol_index + 1
                    ) % len(patrol_points)

            desired_goal = patrol_points[patrol_index]

        elif enemy_state == "chase":
            desired_goal = last_seen_tile

        elif search_mode == "baseline":
            desired_goal = search_anchor

        elif search_mode == "probabilistic":
            desired_goal = belief.most_likely(enemy_tile)

        else:
            #check the last sighting before exploring
            if not enemy_path and enemy_tile == search_anchor:
                checked_anchor = True

            if not checked_anchor:
                desired_goal = search_anchor

            else:
                reached_goal = (
                    not enemy_path
                    and enemy_tile == committed_goal
                )

                if (
                    committed_goal is None
                    or reached_goal
                    or commitment_timer <= 0
                ):
                    committed_goal = belief.most_likely(
                        enemy_tile
                    )
                    commitment_timer = commitment_duration

                desired_goal = committed_goal

        #finish the current step before changing route
        if desired_goal != enemy_goal:
            enemy_goal = desired_goal

            if enemy_path:
                enemy_path = enemy_path[:1]

        if not enemy_path:
            enemy_path = plan_path()

        #follow the route
        movement_left = enemy_speed * dt

        while enemy_path and movement_left > 0:
            target = tile_to_pixel(enemy_path[0])
            offset = target - enemy_position
            distance = offset.length()

            if distance <= movement_left:
                enemy_position = target
                movement_left -= distance
                enemy_path.pop(0)

                if not enemy_path:
                    if pixel_to_tile(enemy_position) != enemy_goal:
                        enemy_path = plan_path()

            else:
                enemy_position += (
                    offset.normalize() * movement_left
                )
                movement_left = 0

        enemy_rect.center = (
            round(enemy_position.x),
            round(enemy_position.y)
        )

        #check capture
        if player_rect.colliderect(enemy_rect):
            finish_round("lost")

        #check orb collection
        if game_status == "playing":
            orb_position = tile_to_pixel(orb_tile)
            orb_rect = pygame.Rect(0, 0, 20, 20)

            orb_rect.center = (
                round(orb_position.x),
                round(orb_position.y)
            )

            if player_rect.colliderect(orb_rect):
                collected_orbs += 1

                if collected_orbs >= orb_target:
                    orb_tile = None
                    finish_round("won")
                else:
                    orb_tile = spawn_orb()

    #draw the floor
    screen.fill(FLOOR_COLOR)

    #draw relative belief strength
    if show_heatmap and enemy_state in ("chase", "search"):
        maximum = max(belief.probabilities.values())

        for tile, probability in belief.probabilities.items():
            strength = (
                probability / maximum if maximum > 0 else 0
            )

            if strength > 0:
                column, row = tile

                pygame.draw.rect(
                    screen,
                    (
                        18 + int(180 * strength),
                        20 + int(95 * strength),
                        24
                    ),
                    pygame.Rect(
                        column * TILE_SIZE,
                        row * TILE_SIZE,
                        TILE_SIZE,
                        TILE_SIZE
                    )
                )

    #draw walls
    for wall in walls:
        pygame.draw.rect(screen, WALL_COLOR, wall)

    #draw the debug route
    if show_path and enemy_path:
        points = [
            (round(enemy_position.x), round(enemy_position.y))
        ]

        for tile in enemy_path:
            point = tile_to_pixel(tile)
            points.append((round(point.x), round(point.y)))

        pygame.draw.lines(screen, PATH_COLOR, False, points, 3)

    #draw the orb
    if orb_tile is not None:
        position = tile_to_pixel(orb_tile)

        pygame.draw.circle(
            screen,
            ORB_COLOR,
            (round(position.x), round(position.y)),
            orb_radius
        )

    #draw characters
    pygame.draw.rect(screen, PLAYER_COLOR, player_rect)
    pygame.draw.rect(screen, ENEMY_COLOR, enemy_rect)

    #apply limited visibility
    if limited_visibility:
        darkness.fill((0, 0, 0, 255))

        pygame.draw.circle(
            darkness,
            (0, 0, 0, 0),
            (round(player_position.x), round(player_position.y)),
            player_view_radius
        )

        screen.blit(darkness, (0, 0))
        draw_orb_hint()

    #draw the status panel
    pygame.draw.rect(screen, (8, 10, 14), (0, 0, WIDTH, 38))

    noise_text = (
        "footstep heard" if heard_flash > 0 else movement_mode
    )

    status = font.render(
        f"orbs: {collected_orbs}/{orb_target}  "
        f"mode: {search_mode}  seed: {round_seed}  "
        f"{noise_text}",
        True,
        TEXT_COLOR
    )

    screen.blit(status, (12, 10))

    #draw stamina
    bar_x = WIDTH - 170

    pygame.draw.rect(
        screen,
        (45, 48, 55),
        (bar_x, 13, 150, 12)
    )

    stamina_color = (
        (190, 85, 65) if exhausted else (100, 190, 140)
    )

    pygame.draw.rect(
        screen,
        stamina_color,
        (bar_x, 13, round(150 * stamina / max_stamina), 12)
    )

    #draw controls
    pygame.draw.rect(
        screen,
        (8, 10, 14),
        (0, HEIGHT - 32, WIDTH, 32)
    )

    controls = font.render(
        "shift: sprint  ctrl: sneak  v: view  h: belief  "
        "p: path  b: mode  r: retry  n: new seed",
        True,
        TEXT_COLOR
    )

    screen.blit(controls, (12, HEIGHT - 25))

    #draw the result
    if game_status != "playing":
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(190)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))

        message = (
            "You escaped!" if game_status == "won"
            else "You were caught!"
        )

        lines = [
            large_font.render(message, True, TEXT_COLOR),
            font.render(
                f"orbs: {collected_orbs}/{orb_target}  "
                f"time: {elapsed_seconds:.1f}s",
                True,
                TEXT_COLOR
            ),
            font.render(log_message, True, TEXT_COLOR),
            font.render(
                "r: retry same seed  n: new seed  escape: quit",
                True,
                TEXT_COLOR
            )
        ]

        for index, text in enumerate(lines):
            screen.blit(
                text,
                text.get_rect(
                    center=(
                        WIDTH // 2,
                        HEIGHT // 2 - 55 + index * 38
                    )
                )
            )

    pygame.display.flip()

pygame.quit()