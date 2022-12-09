; Simple hallway domain for DNNs project

(define (domain simple_domain)

(:requirements :typing)

(:types
    location
)

(:predicates
    (at ?l - location ) ; agent is at location l
    (clear ?l - location ) ; agent is not at location l
    (left ?l1 - location ?l2 - location) ; l1 is left of l2
    (right ?l1 - location ?l2 - location) ; l1 is right of l2
    (above ?l1 - location ?l2 - location) ; l1 is above l2
    (below ?l1 - location ?l2 - location) ; l1 is below l2

    (is-goal ?l - location) ; goal of the agent, used for rendering
    ; auxiliary predicates for non-grounded actions.
    (move_left)
    (move_right)
    (move_up)
    (move_down)
)

; (:actions move_left move_right move_up move_down)

(:action move_left
    :parameters (?l1 - location ?l2 - location)
    :precondition (and 
        (at ?l1) 
        (left ?l2 ?l1)
        (move_left)
    )
    :effect (and (not (at ?l1)) (at ?l2))
)

(:action move_right
    :parameters (?l1 - location ?l2 - location)
    :precondition (and 
        (at ?l1) 
        (left ?l1 ?l2)
        (move_right)
    )
    :effect (and (not (at ?l1)) (at ?l2))
)

(:action move_up
    :parameters (?l1 - location ?l2 - location)
    :precondition (and 
        (at ?l1) 
        (above ?l2 ?l1)
        (move_up)
    )
    :effect (and (not (at ?l1)) (at ?l2))
)

(:action move_down
    :parameters (?l1 - location  ?l2 - location)
    :precondition (and 
        (at ?l1) 
        (above ?l1 ?l2)
        (move_down)
    )
    :effect (and (not (at ?l1)) (at ?l2))
)

)
