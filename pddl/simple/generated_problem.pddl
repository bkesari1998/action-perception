(define (problem simple_problem) (:domain simple_domain)
(:objects
    f0-0f - location
    f0-1f - location
    f0-2f - location
    f1-2f - location
    f2-2f - location
    f3-2f - location
    f4-2f - location
    f5-2f - location
    f5-3f - location
    f5-4f - location
)

(:init
    (move_up)
    (move_down)
    (move_right)
    (move_left)
    (at test)

    (below f0-1f f0-0f)
    (above f0-0f f0-1f)

    (below f0-2f f0-1f)
    (above f0-1f f0-2f)

    (right f1-2f f0-2f)
    (left f0-2f f1-2f)

    (right f2-2f f1-2f)
    (left f1-2f f2-2f)

    (right f3-2f f2-2f)
    (left f2-2f f3-2f)

    (right f4-2f f3-2f)
    (left f3-2f f4-2f)

    (right f5-2f f4-2f)
    (left f4-2f f5-2f)

    (below f5-3f f5-2f)
    (above f5-2f f5-3f)

    (below f5-4f f5-3f)
    (above f5-3f f5-4f)

    (is-goal test)
        )

(:goal (and
    (at test)
    ))
)